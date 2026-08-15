"""Streamlit analytics dashboard for real-time training monitoring."""

from __future__ import annotations
import json; import os; import time
try:
    import streamlit as st; import plotly.graph_objects as go; import numpy as np
except ImportError: pass


def load_training_data(data_dir="data/analytics"):
    result={"summary":{},"learning_curve":[],"pareto_sizes":[],"best_agents":[]}; analytics_path=os.path.join(data_dir,"training_analytics.json")
    if os.path.exists(analytics_path):
        try:
            with open(analytics_path) as f: data=json.load(f); result["summary"]=data.get("summary",{}); result["learning_curve"]=data.get("learning_curve",[])
            result["best_agents"]=data.get("best_agents",[]); result["pareto_sizes"]=[{"generation":i,"pareto_front_size":r.get("pareto_front_size",0)} for i,r in enumerate(data.get("learning_curve",[]))]
        except: pass
    return result

@st.cache_data(ttl=5)
def get_cached_metrics(data_dir="data/analytics"): return load_training_data(data_dir)


def create_learning_chart(lc):
    if not lc: fig=go.Figure(); fig.add_annotation(text="No data",xref="paper",yref="paper",showarrow=False,x=0.5,y=0.5); return fig
    fig=go.Figure()
    fig.add_trace(go.Scatter(x=[d["generation"] for d in lc],y=[d["raw_fitness"] for d in lc],name="Raw Fitness",mode="lines",line=dict(width=1,color="rgba(100,149,237,0.5)")))
    fig.add_trace(go.Scatter(x=[d["generation"] for d in lc],y=[d["smoothed_fitness"] for d in lc],name="Smoothed (EMA)",mode="lines",line=dict(width=3,color="#1f77b4")))
    fig.update_layout(title="Learning Curve — Agent Fitness Over Generations",xaxis_title="Generation",yaxis_title="Fitness Score",height=400)
    return fig

def create_pareto_chart(pd):
    if not pd: fig=go.Figure(); fig.add_annotation(text="No data",xref="paper",yref="paper",showarrow=False,x=0.5,y=0.5); return fig
    fig=go.Figure(); fig.add_trace(go.Scatter(x=[d["generation"] for d in pd],y=[d["pareto_front_size"] for d in pd],name="Pareto Size",mode="lines+markers"))
    fig.update_layout(title="Pareto Front Size Evolution",xaxis_title="Generation",yaxis_title="Number of Pareto Members",height=350)
    return fig

def main():
    st.set_page_config(page_title="KSP Neuroevolution Dashboard",page_icon="\U0001f680",layout="wide")
    with st.sidebar: st.title("\u2699\ufe0f Config"); data_dir=st.text_input("Analytics dir","data/analytics"); refresh=st.slider("Auto-refresh (sec)",1,30,5); auto_refresh=st.checkbox("Enable auto-refresh",value=True)

    st.title("\U0001f680 KSP Neuroevolution Analytics Dashboard")
    with st.spinner("Loading analytics data..."): metrics=get_cached_metrics(data_dir)
    summary=metrics.get("summary",{}); status=summary.get("status","no_data")
    c1,c2,c3,c4=st.columns(4)
    with c1: st.metric("Status",status.upper())
    with c2: st.metric("Generations",summary.get("total_generations",0))
    with c3: st.metric("Best Fitness Ever",f"{summary.get('best_fitness_overall',0):.4f}")
    with c4: 
        el=summary.get("time_elapsed_sec",0); m,s=int(el//60),int(el%60)
        st.metric("Time Elapsed",f"{m}m {s}s")

    st.subheader("\U0001f4c8 Training Progress"); c1,c2=st.columns(2)
    with c1: st.plotly_chart(create_learning_chart(metrics.get("learning_curve",[])),use_container_width=True)
    with c2: st.plotly_chart(create_pareto_chart(metrics.get("pareto_sizes",[])[:500]),use_container_width=True)

    st.subheader("\U0001f9ea Population Health"); best=metrics.get("best_agents",[])
    if best:
        cols=st.columns([2,1,1,1])
        with cols[0]: st.markdown("**Generation**")
        with cols[1]: st.markdown("**Best Fitness**")
        with cols[2]: st.markdown("**Mean Fitness**")
        with cols[3]: st.markdown("**Timestamp**")
        for a in best[-5:]:
            ts=a.get("timestamp",0); tstr=time.strftime("%H:%M:%S",time.localtime(ts)) if ts else "N/A"
            cols=st.columns([2,1,1,1])
            with cols[0]: st.text(str(a.get("generation","?")))
            with cols[1]: st.text(f"{a.get('best_fitness',0):.4f}")
            with cols[2]: st.text(f"{a.get('mean_fitness',0):.4f}")
            with cols[3]: st.text(tstr)

    st.subheader("\U0001f50c Connection Status")
    try:
        from ksp_neuro.ksp_interface.hardened import check_ksp_health; kh=check_ksp_health()
        if kh.get("reachable"): st.success(f"\u2705 KSP reachable — Latency: {kh['latency_ms']:.1f}ms")
        else: st.warning(f"\u26a0\ufe0f KSP not reachable — {kh.get('error','Unknown')}")
    except Exception as e: st.info(str(e))

    if auto_refresh and refresh>0:
        while True: time.sleep(refresh); st.rerun()


if __name__=="__main__": main()
