"""Production-grade kRPC interface with connection pooling, auto-reconnect, and simulation fallback."""

from __future__ import annotations
import time; import math; import socket; import copy; import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class VesselState:
    name:str=""; altitude_agl:float=0.0; altitude_asl:float=0.0; speed_surface:float=0.0; speed_orbital:float=0.0
    prograde_velocity:list=field(default_factory=lambda:[0.,0.,0.]); radial_velocity:list=field(default_factory=lambda:[0.,0.,0.])
    normal_velocity:list=field(default_factory=lambda:[0.,0.,0.]); semi_major_axis:float=0.0; eccentricity:float=0.0
    apoapsis_alt:float=0.0; periapsis_alt:float=0.0; mass:float=0.0; fuel_total:float=0.0; throttle:float=0.0

@dataclass
class CelestialBodyInfo:
    name:str="Kerbin"; radius:float=600_000.0; mu:float=8.7129645463e12; atmosphere:bool=True
    atm_scale_height:float=5_000.0; atm_surface_density:float=1.225

@dataclass
class KSPGameState:
    vessel:VesselState=field(default_factory=VesselState); body:CelestialBodyInfo=field(default_factory=CelestialBodyInfo)
    time:float=0.0; phase:str="PRELAUNCH"; connected:bool=False


class kRPCConnectionPool:
    def __init__(self, host="localhost", port=5000):
        self.host = host; self.port = port; self._connection = None; self._is_connected = False
        self.reconnect_count = 0; self.error_count = 0
    @property
    def is_connected(self): return self._is_connected and self._connection is not None
    def connect(self):
        try:
            import krpc as _krpc  # type: ignore
            if self._connection is None:
                self._connection = _krpc.connect(name="ksp-neuroevolution", address=self.host, port=self.port)
            self._is_connected = True; self.reconnect_count += 1; return True
        except Exception as e:
            logger.warning(f"kRPC connection failed ({e})"); self.error_count += 1; self._is_connected = False; return False
    def disconnect(self):
        if self._connection is not None:
            try: self._connection.close()
            except: pass
            self._connection=None
        self._is_connected=False
    def ensure_connected(self):
        if self.is_connected: 
            try: game=self._connection.game; return game is not None
            except: pass
        import krpc as _krpc  # type: ignore
        try: self._connection=_krpc.connect(name="ksp-neuroevolution",address=self.host,port=self.port)
        except Exception: 
            self.error_count+=1; return False
        self._is_connected=True; self.reconnect_count+=1; return True


class KSPInterface:
    def __init__(self,host="localhost",port=5000): self.pool=kRPCConnectionPool(host=host,port=port); self.use_fallback=False
    @property
    def is_connected(self): return self.pool.is_connected and not self.use_fallback
    @property
    def connection_status(self): return {"connected":self.is_connected,"using_fallback":self.use_fallback,"reconnect_count":self.pool.reconnect_count,"error_count":self.pool.error_count}
    def connect(self): return self.pool.connect()
    def disconnect(self): self.pool.disconnect(); self.use_fallback=True; logger.info("Disconnected from KSP, enabling simulation fallback")
    def get_vessel_state(self) -> KSPGameState:
        if not self.is_connected and self.use_fallback: return KSPGameState(connected=False); 
        if not self.pool.ensure_connected(): return KSPGameState(connected=False)
        try:
            import krpc  # type: ignore
            s=self.pool._connection.space_center; v=s.active_vessel; st=KSPGameState(); st.connected=True
            st.vessel.name=v.name if hasattr(v,'name') else "Unknown"
            st.vessel.altitude_asl=float(getattr(v.orbit,'radius',0)-getattr(v.orbit.body,'radius',600_000))
            st.vessel.speed_orbital=float(v.orbit.speed) if hasattr(v,'orbit') and v.orbit else 0.0
            return st
        except Exception as e: logger.error(f"Failed to read KSP state: {e}"); self.use_fallback=True; return KSPGameState(connected=False)

    def get_observation(self):
        state=self.get_vessel_state(); body=state.body if state.body else CelestialBodyInfo()
        r=max(body.radius+max(state.vessel.altitude_asl,0),1); ov=math.sqrt(body.mu/r)
        return [state.vessel.altitude_asl/(body.soI_radius*0.5) if hasattr(body,'soI_radius') else state.vessel.altitude_asl/42_079_873,
                state.vessel.speed_orbital/max(ov*2,1),state.vessel.prograde_velocity[0]/max(ov,1) if ov>0 else 0,
                state.vessel.radial_velocity[0]/max(ov,1) if ov>0 else 0,state.vessel.normal_velocity[0]/max(ov,1) if ov>0 else 0,
                (body.radius+state.vessel.altitude_asl)/body.soI_radius if hasattr(body,'soI_radius') else 0.5]


class SimulationFallback:
    def __init__(self): self.active=False; self.simulator=None; self.switch_count=0
    def enable(self,task_name="vertical_burn",n_agents=100):
        from ksp_neuro.sim_env.vectorized_sim import VectorizedOrbitSimulator
        self.active=True; self.switch_count+=1
        if self.simulator is None or self.simulator.task_name!=task_name: self.simulator=VectorizedOrbitSimulator(task_name=task_name,n_agents=n_agents)
        return True
    def disable(self): self.active=False; return True


def check_ksp_health(host="localhost", port=5000, timeout=2.0):
    result = {"reachable": False, "latency_ms": None, "error": None}
    start = time.time()
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM); sock.settimeout(timeout)
        rc = sock.connect_ex((host, port)); elapsed = (time.time() - start) * 1000
        result["reachable"] = rc == 0; result["latency_ms"] = round(elapsed, 2); sock.close()
    except socket.timeout:
        result["error"] = "Connection timed out"
    except Exception as e:
        result["error"] = str(e)
    return result

def get_connection_status(interface,health_check=True):
    status=interface.connection_status
    if health_check and not status["connected"]: status["health"]=check_ksp_health(interface.pool.host,int(interface.pool.port))
    return status
