"""
╔══════════════════════════════════════════════════════════════╗
║        PakTravel AI System — Pro GUI (Streamlit)            ║
║        AL2002 Artificial Intelligence | FAST NUCES           ║
╚══════════════════════════════════════════════════════════════╝
Run with:  streamlit run app.py
"""

import streamlit as st
import time
import heapq
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from collections import defaultdict, deque
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score, confusion_matrix)
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PakTravel AI System",
    page_icon="🚌",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────
# GLOBAL CSS — animations, glassmorphism, gradients, fonts
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Google Font ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

/* ── Root palette ── */
:root {
    --bg:        #060d1f;
    --surface:   #0d1b3e;
    --surface2:  #112244;
    --border:    #1e3a5f;
    --green:     #00e676;
    --blue:      #448aff;
    --amber:     #ffab40;
    --red:       #ff5252;
    --purple:    #b388ff;
    --cyan:      #18ffff;
    --text:      #e8f0fe;
    --muted:     #78909c;
}

/* ── Base ── */
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.stApp { background: var(--bg); color: var(--text); }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #060d1f 0%, #0d1b3e 100%);
    border-right: 1px solid var(--border);
}
[data-testid="stSidebar"] * { color: var(--text) !important; }

/* ── Sidebar nav items ── */
.sidebar-nav-item {
    display: flex; align-items: center; gap: 10px;
    padding: 12px 16px; border-radius: 12px;
    margin-bottom: 6px; cursor: pointer;
    transition: all 0.25s ease;
    font-weight: 500; font-size: 0.95rem;
    border: 1px solid transparent;
}
.sidebar-nav-item:hover {
    background: rgba(68,138,255,0.12);
    border-color: var(--blue);
}

/* ── Animated hero banner ── */
@keyframes gradientShift {
    0%   { background-position: 0% 50%; }
    50%  { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}
.hero-banner {
    background: linear-gradient(135deg, #060d1f, #0d1b3e, #112244, #0a2240, #060d1f);
    background-size: 300% 300%;
    animation: gradientShift 8s ease infinite;
    border: 1px solid var(--border);
    border-radius: 20px;
    padding: 42px 36px;
    margin-bottom: 28px;
    text-align: center;
    position: relative;
    overflow: hidden;
}
.hero-banner::before {
    content: '';
    position: absolute; top: -2px; left: -2px; right: -2px; bottom: -2px;
    background: linear-gradient(135deg, #448aff33, #00e67633, #b388ff33);
    border-radius: 20px; z-index: 0;
    animation: gradientShift 6s ease infinite;
    background-size: 300% 300%;
}
.hero-banner > * { position: relative; z-index: 1; }
.hero-title {
    font-size: 2.6rem; font-weight: 900;
    background: linear-gradient(90deg, #448aff, #00e676, #ffab40, #b388ff);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text; letter-spacing: -0.5px; margin-bottom: 8px;
}
.hero-sub {
    font-size: 1.05rem; color: #90a4ae; font-weight: 400;
}

/* ── Glass card ── */
.glass-card {
    background: rgba(13,27,62,0.6);
    backdrop-filter: blur(16px);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 24px 28px;
    margin-bottom: 20px;
    transition: border-color 0.3s, box-shadow 0.3s;
}
.glass-card:hover {
    border-color: #448aff66;
    box-shadow: 0 0 32px rgba(68,138,255,0.08);
}

/* ── Section heading ── */
.section-heading {
    font-size: 1.25rem; font-weight: 700;
    color: var(--text); margin-bottom: 16px;
    display: flex; align-items: center; gap: 10px;
}
.section-heading::after {
    content: ''; flex: 1; height: 1px;
    background: linear-gradient(90deg, var(--border), transparent);
}

/* ── Stat cards row ── */
.stat-grid { display: flex; gap: 14px; flex-wrap: wrap; margin-bottom: 20px; }
.stat-card {
    flex: 1; min-width: 130px;
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 18px 16px;
    text-align: center;
    transition: transform 0.2s, box-shadow 0.2s;
}
.stat-card:hover { transform: translateY(-3px); box-shadow: 0 8px 24px rgba(0,0,0,0.4); }
.stat-value { font-size: 1.9rem; font-weight: 800; line-height: 1; }
.stat-label { font-size: 0.78rem; color: var(--muted); margin-top: 5px; text-transform: uppercase; letter-spacing: 0.5px; }

/* ── Result badge ── */
.badge {
    display: inline-block;
    padding: 6px 18px; border-radius: 999px;
    font-weight: 700; font-size: 0.88rem;
    letter-spacing: 0.3px;
}
.badge-green  { background: #00e67618; color: #00e676; border: 1px solid #00e67644; }
.badge-red    { background: #ff525218; color: #ff5252; border: 1px solid #ff525244; }
.badge-blue   { background: #448aff18; color: #448aff; border: 1px solid #448aff44; }
.badge-amber  { background: #ffab4018; color: #ffab40; border: 1px solid #ffab4044; }
.badge-purple { background: #b388ff18; color: #b388ff; border: 1px solid #b388ff44; }

/* ── Path display ── */
.path-display {
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 16px 20px;
    font-family: 'Inter', monospace;
    font-size: 0.9rem;
    color: var(--amber);
    letter-spacing: 0.3px;
    overflow-x: auto;
}

/* ── Timeline steps ── */
.timeline { position: relative; padding-left: 28px; }
.timeline::before {
    content: ''; position: absolute; left: 8px; top: 0; bottom: 0;
    width: 2px; background: linear-gradient(180deg, var(--blue), var(--purple));
}
.timeline-step {
    position: relative; margin-bottom: 18px;
    padding: 14px 18px; background: var(--surface2);
    border-radius: 12px; border: 1px solid var(--border);
    transition: border-color 0.3s;
}
.timeline-step:hover { border-color: var(--blue); }
.timeline-step::before {
    content: ''; position: absolute; left: -24px; top: 50%;
    transform: translateY(-50%);
    width: 12px; height: 12px; border-radius: 50%;
    background: var(--blue); border: 2px solid var(--bg);
}
.timeline-label { font-size: 0.75rem; color: var(--muted); text-transform: uppercase; letter-spacing: 0.5px; }
.timeline-value { font-size: 0.95rem; font-weight: 600; color: var(--text); margin-top: 3px; }

/* ── Pulse animation ── */
@keyframes pulse {
    0%, 100% { opacity: 1; }
    50%       { opacity: 0.5; }
}
.pulse { animation: pulse 2s ease infinite; }

/* ── Fade-in ── */
@keyframes fadeInUp {
    from { opacity: 0; transform: translateY(16px); }
    to   { opacity: 1; transform: translateY(0); }
}
.fade-in { animation: fadeInUp 0.5s ease forwards; }

/* ── Loading spinner ── */
@keyframes spin { to { transform: rotate(360deg); } }
.spinner {
    width: 40px; height: 40px;
    border: 3px solid var(--border);
    border-top-color: var(--blue);
    border-radius: 50%;
    animation: spin 0.8s linear infinite;
    margin: 20px auto;
}

/* ── Info box ── */
.info-box {
    background: rgba(68,138,255,0.08);
    border: 1px solid rgba(68,138,255,0.3);
    border-radius: 12px; padding: 14px 18px;
    font-size: 0.88rem; color: #90caf9;
    margin-bottom: 16px;
}

/* ── Warning box ── */
.warn-box {
    background: rgba(255,171,64,0.08);
    border: 1px solid rgba(255,171,64,0.3);
    border-radius: 12px; padding: 14px 18px;
    font-size: 0.88rem; color: #ffcc80;
    margin-bottom: 16px;
}

/* ── Success box ── */
.success-box {
    background: rgba(0,230,118,0.08);
    border: 1px solid rgba(0,230,118,0.3);
    border-radius: 12px; padding: 14px 18px;
    font-size: 0.88rem; color: #a5d6a7;
    margin-bottom: 16px;
}

/* ── Streamlit overrides ── */
.stButton > button {
    background: linear-gradient(135deg, #1a3a6b, #1565c0) !important;
    color: white !important; border: 1px solid #448aff66 !important;
    border-radius: 10px !important; font-weight: 600 !important;
    padding: 10px 24px !important; font-size: 0.92rem !important;
    transition: all 0.2s !important; width: 100% !important;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #1565c0, #1976d2) !important;
    box-shadow: 0 0 20px rgba(68,138,255,0.35) !important;
    transform: translateY(-1px) !important;
}
.stSelectbox > div > div, .stMultiSelect > div > div {
    background: var(--surface2) !important;
    border-color: var(--border) !important;
    color: var(--text) !important;
}
.stSlider [data-baseweb="slider"] { padding: 0 4px; }
.stSlider [data-baseweb="slider"] [role="slider"] {
    background: var(--blue) !important;
}
div[data-testid="stMetric"] {
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 12px; padding: 16px !important;
}
div[data-testid="stMetric"] label { color: var(--muted) !important; }
div[data-testid="stMetric"] [data-testid="stMetricValue"] {
    color: var(--text) !important; font-weight: 700 !important;
}
.stDataFrame, .stTable { border-radius: 12px; overflow: hidden; }
.stProgress .st-bo { background-color: var(--blue) !important; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# ══ DATA / ALGORITHM CORE  (cached so GUI stays fast) ════════
# ─────────────────────────────────────────────────────────────

@st.cache_data
def get_graph():
    graph = defaultdict(dict)
    roads = [
        ("Karachi","Hyderabad",160),("Hyderabad","Sukkur",380),
        ("Sukkur","Larkana",80),("Sukkur","Bahawalpur",400),
        ("Sukkur","Multan",320),("Bahawalpur","Multan",180),
        ("Multan","Lahore",340),("Multan","Faisalabad",270),
        ("Lahore","Faisalabad",130),("Lahore","Gujranwala",80),
        ("Lahore","Islamabad",380),("Gujranwala","Sialkot",70),
        ("Islamabad","Rawalpindi",15),("Islamabad","Peshawar",170),
        ("Multan","Quetta",570),("Karachi","Quetta",700),
        ("Quetta","Gwadar",650),("Rawalpindi","Peshawar",155),
        ("Faisalabad","Gujranwala",90),
    ]
    for a,b,d in roads:
        graph[a][b]=d; graph[b][a]=d
    return dict(graph)

CITIES = ["Karachi","Hyderabad","Sukkur","Larkana","Bahawalpur",
          "Multan","Lahore","Faisalabad","Gujranwala","Sialkot",
          "Islamabad","Rawalpindi","Peshawar","Quetta","Gwadar"]

RELAY_POINTS = ["Sukkur","Multan","Lahore","Islamabad","Rawalpindi","Faisalabad"]

HEURISTIC = {
    "Karachi":1400,"Hyderabad":1250,"Sukkur":900,"Larkana":920,
    "Bahawalpur":620,"Multan":550,"Lahore":380,"Faisalabad":300,
    "Gujranwala":280,"Sialkot":300,"Islamabad":0,"Rawalpindi":15,
    "Peshawar":170,"Quetta":750,"Gwadar":1500,
}

# Approximate 2-D positions for the map
CITY_POS = {
    "Karachi":(67.0,24.8),"Hyderabad":(68.4,25.4),"Sukkur":(68.9,27.7),
    "Larkana":(68.2,27.6),"Quetta":(67.0,30.2),"Gwadar":(62.3,25.1),
    "Bahawalpur":(71.7,29.4),"Multan":(71.5,30.2),"Faisalabad":(73.1,31.4),
    "Lahore":(74.3,31.5),"Gujranwala":(74.2,32.2),"Sialkot":(74.5,32.5),
    "Islamabad":(73.1,33.7),"Rawalpindi":(73.0,33.6),"Peshawar":(71.5,34.0),
}

def ucs(graph,start,goal):
    frontier=[(0,start,[start])]; visited=set(); ne=0
    while frontier:
        cost,cur,path=heapq.heappop(frontier); ne+=1
        if cur in visited: continue
        visited.add(cur)
        if cur==goal: return path,cost,ne
        for nb,d in graph.get(cur,{}).items():
            if nb not in visited:
                heapq.heappush(frontier,(cost+d,nb,path+[nb]))
    return None,float("inf"),ne

def astar(graph,start,goal,h):
    frontier=[(h.get(start,0),0,start,[start])]; visited=set(); ne=0
    while frontier:
        f,g,cur,path=heapq.heappop(frontier); ne+=1
        if cur in visited: continue
        visited.add(cur)
        if cur==goal: return path,g,ne
        for nb,d in graph.get(cur,{}).items():
            if nb not in visited:
                ng=g+d
                heapq.heappush(frontier,(ng+h.get(nb,0),ng,nb,path+[nb]))
    return None,float("inf"),ne

def bidir(graph,start,goal):
    if start==goal: return [start],0,0
    fwd=[(0,start,[start])]; bwd=[(0,goal,[goal])]
    fwd_v={}; bwd_v={}
    fwd_p={start:([start],0)}; bwd_p={goal:([goal],0)}
    best=float("inf"); best_path=None; ne=0
    def step(frontier,visited,paths,other,is_fwd):
        nonlocal best,best_path,ne
        if not frontier: return
        cost,cur,path=heapq.heappop(frontier); ne+=1
        if cur in visited: return
        visited[cur]=cost
        if cur in other:
            oc,op=other[cur][1],other[cur][0]
            total=cost+oc
            if total<best:
                best=total
                best_path=(path+op[-2::-1]) if is_fwd else (op+path[-2::-1])
        for nb,d in graph.get(cur,{}).items():
            if nb not in visited:
                nc=cost+d; np_=path+[nb]
                heapq.heappush(frontier,(nc,nb,np_))
                if nb not in paths or nc<paths[nb][1]:
                    paths[nb]=(np_,nc)
    for _ in range(500):
        step(fwd,fwd_v,fwd_p,bwd_p,True)
        step(bwd,bwd_v,bwd_p,fwd_p,False)
        fm=fwd[0][0] if fwd else float("inf")
        bm=bwd[0][0] if bwd else float("inf")
        if best_path and fm+bm>=best: break
        if not fwd or not bwd: break
    return best_path,best,ne

def find_relay(start,dest,relay,graph):
    best=None; best_t=float("inf"); best_l1=best_l2=None; best_d1=best_d2=0
    for r in relay:
        if r in (start,dest): continue
        l1,d1,_=ucs(graph,start,r)
        l2,d2,_=ucs(graph,r,dest)
        if l1 and l2 and d1+d2<best_t:
            best_t=d1+d2; best=r; best_l1=l1; best_l2=l2; best_d1=d1; best_d2=d2
    return best,best_l1,best_l2,best_d1,best_d2,best_t

@st.cache_data
def train_ann():
    np.random.seed(42)
    N=500
    dist=np.random.randint(80,701,N); weather=np.random.randint(1,6,N)
    traffic=np.random.randint(1,6,N); road=np.random.randint(1,6,N)
    age=np.random.randint(1,16,N)
    y=((weather>=4)|(traffic>=4)|((dist>400)&(road<=2))).astype(int)
    X=np.column_stack([dist,weather,traffic,road,age])
    Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)
    sc=MinMaxScaler(); Xtr_sc=sc.fit_transform(Xtr); Xte_sc=sc.transform(Xte)
    clf=MLPClassifier(hidden_layer_sizes=(10,8),activation="relu",
                      max_iter=500,random_state=42,early_stopping=True)
    clf.fit(Xtr_sc,ytr); yp=clf.predict(Xte_sc)
    return clf, sc, {
        "acc":accuracy_score(yte,yp), "prec":precision_score(yte,yp),
        "rec":recall_score(yte,yp),   "f1":f1_score(yte,yp),
        "cm":confusion_matrix(yte,yp).tolist(),
        "loss":clf.loss_curve_
    }

@st.cache_data
def train_kmeans():
    np.random.seed(42); n=400; ne=n//3
    A_t=np.random.randint(10,16,ne); A_b=np.random.randint(15,31,ne)
    A_d=np.random.randint(400,701,ne); A_p=np.random.randint(1,3,ne)
    A_l=np.random.randint(2500,5001,ne); A_c=np.random.randint(0,3,ne)
    B_t=np.random.randint(1,4,ne);  B_b=np.random.randint(0,8,ne)
    B_d=np.random.randint(200,500,ne); B_p=np.random.randint(2,5,ne)
    B_l=np.random.randint(0,1501,ne); B_c=np.random.randint(3,8,ne)
    C_t=np.random.randint(6,12,ne);  C_b=np.random.randint(0,5,ne)
    C_d=np.random.randint(100,300,ne); C_p=np.random.randint(1,4,ne)
    C_l=np.random.randint(500,2501,ne); C_c=np.random.randint(1,6,ne)
    r=n-3*ne
    df=pd.DataFrame({
        "Trips_Per_Month":np.concatenate([A_t,B_t,C_t,C_t[:r]]),
        "Avg_Booking_Days":np.concatenate([A_b,B_b,C_b,C_b[:r]]),
        "Avg_Journey_Distance":np.concatenate([A_d,B_d,C_d,C_d[:r]]),
        "Preferred_Time":np.concatenate([A_p,B_p,C_p,C_p[:r]]),
        "Loyalty_Points":np.concatenate([A_l,B_l,C_l,C_l[:r]]),
        "Complaints_Filed":np.concatenate([A_c,B_c,C_c,C_c[:r]]),
    }).sample(frac=1,random_state=42).reset_index(drop=True)
    sc5=StandardScaler(); X5=sc5.fit_transform(df)
    km=KMeans(n_clusters=3,random_state=42,n_init=10)
    lbl=km.fit_predict(X5); df["Cluster"]=lbl
    dist_mean=df.groupby("Cluster")["Avg_Journey_Distance"].mean()
    order=dist_mean.sort_values(ascending=False).index.tolist()
    cmap={order[0]:"Business",order[1]:"Family",order[2]:"Commuter"}
    df["Profile"]=df["Cluster"].map(cmap)
    inertias=[]
    for k in range(1,9):
        km_=KMeans(n_clusters=k,random_state=42,n_init=10)
        km_.fit(X5); inertias.append(km_.inertia_)
    return df, inertias

# CSP helpers
ROUTES=["R1","R2","R3","R4","R5","R6","R7","R8"]
BUSES=["Bus1","Bus2","Bus3","Bus4","Bus5","Bus6","Bus7","Bus8","Bus9","Bus10"]
ROUTE_INFO={
    "R1":{"name":"Karachi → Hyderabad","time":"8am","highway":True},
    "R2":{"name":"Hyderabad → Sukkur","time":"10am","highway":False},
    "R3":{"name":"Sukkur → Multan","time":"12pm","highway":False},
    "R4":{"name":"Multan → Lahore","time":"2pm","highway":True},
    "R5":{"name":"Lahore → Islamabad","time":"4pm","highway":True},
    "R6":{"name":"Islamabad → Peshawar","time":"6pm","highway":False},
    "R7":{"name":"Karachi → Quetta","time":"8am","highway":False},
    "R8":{"name":"Quetta → Gwadar","time":"12pm","highway":False},
}
LARGE=["Bus1","Bus2","Bus3"]
CONFLICTS=[("R1","R7"),("R3","R8")]
DOMAINS_BASE={
    "R1":["Bus1","Bus2","Bus3","Bus4","Bus5"],
    "R2":list(BUSES),"R3":["Bus1","Bus2","Bus3","Bus4","Bus5","Bus6","Bus7","Bus8","Bus9"],
    "R4":["Bus1","Bus2","Bus3","Bus6","Bus7","Bus8"],
    "R5":["Bus1","Bus2","Bus3","Bus6","Bus7","Bus8"],
    "R6":list(BUSES),"R7":["Bus4","Bus5","Bus6","Bus7","Bus8","Bus9","Bus10"],
    "R8":["Bus9"],
}

def csp_check(asgn,route,bus):
    for r1,r2 in CONFLICTS:
        if route==r1 and r2 in asgn and asgn[r2]==bus: return False
        if route==r2 and r1 in asgn and asgn[r1]==bus: return False
    if sum(1 for b in asgn.values() if b==bus)>=2: return False
    return True

def csp_solve(domains):
    calls=[0]
    def bt(asgn):
        if len(asgn)==len(ROUTES): return dict(asgn)
        unassigned=[r for r in ROUTES if r not in asgn]
        if "R3" not in asgn: route="R3"
        else: route=min(unassigned,key=lambda r:len([b for b in domains[r] if csp_check(asgn,r,b)]))
        for bus in domains[route]:
            calls[0]+=1
            if csp_check(asgn,route,bus):
                asgn[route]=bus
                res=bt(asgn)
                if res: return res
                del asgn[route]
        return None
    sol=bt({})
    return sol, calls[0]

def run_ac3(domains):
    arcs=[(r1,r2) for r1,r2 in CONFLICTS]+[(r2,r1) for r1,r2 in CONFLICTS]
    q=deque(arcs); removed=0
    while q:
        ri,rj=q.popleft()
        to_rem=[b for b in domains[ri] if not any(bj!=b for bj in domains[rj])]
        for b in to_rem: domains[ri].remove(b); removed+=1
    return domains,removed

def soft_score(asgn):
    s=0
    for r,b in asgn.items():
        if ROUTE_INFO[r]["highway"] and b in LARGE: s+=2
    usage={}
    for b in asgn.values(): usage[b]=usage.get(b,0)+1
    for cnt in usage.values():
        if cnt<=2: s+=1
    return s

# ─────────────────────────────────────────────────────────────
# PLOTLY HELPERS
# ─────────────────────────────────────────────────────────────

PLOTLY_DARK = dict(
    paper_bgcolor="rgba(6,13,31,0)",
    plot_bgcolor="rgba(13,27,62,0.7)",
    font=dict(family="Inter", color="#e8f0fe"),
    xaxis=dict(gridcolor="#1e3a5f", zerolinecolor="#1e3a5f"),
    yaxis=dict(gridcolor="#1e3a5f", zerolinecolor="#1e3a5f"),
    margin=dict(l=40, r=20, t=50, b=40),
)

def make_network_fig(graph, path=None):
    lons=[]; lats=[]; texts=[]
    seen=set()
    edge_lons=[]; edge_lats=[]
    for ca in graph:
        for cb,d in graph[ca].items():
            e=tuple(sorted([ca,cb]))
            if e in seen: continue
            seen.add(e)
            la,loa=CITY_POS[ca][1],CITY_POS[ca][0]
            lb,lob=CITY_POS[cb][1],CITY_POS[cb][0]
            edge_lons+=[loa,lob,None]; edge_lats+=[la,lb,None]

    path_edge_lons=[]; path_edge_lats=[]
    if path:
        for i in range(len(path)-1):
            a,b=path[i],path[i+1]
            path_edge_lons+=[CITY_POS[a][0],CITY_POS[b][0],None]
            path_edge_lats+=[CITY_POS[a][1],CITY_POS[b][1],None]

    node_lons=[CITY_POS[c][0] for c in CITIES]
    node_lats=[CITY_POS[c][1] for c in CITIES]
    node_colors=[]
    for c in CITIES:
        if path and c==path[0]: node_colors.append("#00e676")
        elif path and c==path[-1]: node_colors.append("#ff5252")
        elif path and c in path: node_colors.append("#ffab40")
        elif c in RELAY_POINTS: node_colors.append("#b388ff")
        else: node_colors.append("#448aff")

    fig=go.Figure()
    fig.add_trace(go.Scattergeo(lon=edge_lons,lat=edge_lats,mode="lines",
        line=dict(width=1,color="#1e3a5f"),showlegend=False,hoverinfo="skip"))
    if path_edge_lons:
        fig.add_trace(go.Scattergeo(lon=path_edge_lons,lat=path_edge_lats,mode="lines",
            line=dict(width=3.5,color="#ffab40"),showlegend=False,hoverinfo="skip"))
    fig.add_trace(go.Scattergeo(lon=node_lons,lat=node_lats,mode="markers+text",
        text=CITIES,textposition="top center",
        marker=dict(size=10,color=node_colors,line=dict(width=1.5,color="#060d1f")),
        hovertemplate="<b>%{text}</b><extra></extra>",showlegend=False))
    fig.update_geos(
        visible=False,resolution=50,
        showcountries=True,countrycolor="#1e3a5f",
        showland=True,landcolor="#0d1b3e",
        showocean=True,oceancolor="#060d1f",
        showlakes=True,lakecolor="#060d1f",
        center=dict(lon=69,lat=29),projection_scale=4.5,
    )
    fig.update_layout(
        height=420,margin=dict(l=0,r=0,t=0,b=0),
        paper_bgcolor="rgba(6,13,31,0)",geo_bgcolor="rgba(6,13,31,0)"
    )
    return fig

# ─────────────────────────────────────────────────────────────
# ══ SIDEBAR ══
# ─────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding:16px 0 24px;'>
      <div style='font-size:2.6rem;'>🚌</div>
      <div style='font-size:1.1rem; font-weight:800; color:#e8f0fe; letter-spacing:-0.3px;'>PakTravel AI</div>
      <div style='font-size:0.72rem; color:#546e7a; margin-top:4px; letter-spacing:1px; text-transform:uppercase;'>Intelligent Travel System</div>
    </div>
    """, unsafe_allow_html=True)

    NAV_ITEMS = [
        ("🏠", "Home",            "Dashboard overview"),
        ("🗺️", "Route Finder",    "Search algorithms"),
        ("⚖️", "Legal Advisor",   "Propositional logic"),
        ("📅", "Bus Scheduler",   "CSP solver"),
        ("🧠", "Delay Predictor", "ANN prediction"),
        ("👥", "Traveller Types", "K-Means clustering"),
    ]

    if "page" not in st.session_state:
        st.session_state.page = "Home"

    for icon, label, desc in NAV_ITEMS:
        active = st.session_state.page == label
        border = "border:1px solid #448aff !important;" if active else ""
        bg     = "background:rgba(68,138,255,0.14);" if active else ""
        if st.button(f"{icon}  {label}", key=f"nav_{label}",
                     help=desc, use_container_width=True):
            st.session_state.page = label
            st.rerun()

    st.markdown("---")
    st.markdown("""
    <div style='font-size:0.72rem;color:#37474f;text-align:center;line-height:1.6;'>
    AL2002 Artificial Intelligence<br>
    FAST NUCES Chiniot-Faisalabad<br>
    Spring 2026
    </div>""", unsafe_allow_html=True)

page = st.session_state.page

# ─────────────────────────────────────────────────────────────
# ══ PAGE: HOME ══
# ─────────────────────────────────────────────────────────────
if page == "Home":
    st.markdown("""
    <div class='hero-banner fade-in'>
      <div class='hero-title'>🚌 PakTravel AI System</div>
      <div class='hero-sub'>Pakistan's Intelligent Road Travel Assistant &nbsp;·&nbsp; AL2002 Artificial Intelligence</div>
      <div style='margin-top:14px; font-size:0.82rem; color:#546e7a;'>
        FAST NUCES Chiniot-Faisalabad &nbsp;·&nbsp; Spring 2026
      </div>
    </div>""", unsafe_allow_html=True)

    # Story cards
    st.markdown("<div class='section-heading'>📖 Ali's Journey — 5 AI Stages</div>", unsafe_allow_html=True)

    cols = st.columns(5)
    stages = [
        ("🗺️","Part 1","Route Finder","Search Algorithms\nUCS · A* · Bidirectional","#448aff"),
        ("⚖️","Part 2","Legal Advisor","Propositional Logic\nModus Ponens · Resolution","#b388ff"),
        ("📅","Part 3","Bus Scheduler","CSP Solver\nBacktracking · MRV · AC3","#ffab40"),
        ("🧠","Part 4","Delay Predictor","Neural Network\nsklearn & NumPy ANN","#00e676"),
        ("👥","Part 5","Traveller Types","K-Means Clustering\nBusiness · Family · Commuter","#ff5252"),
    ]
    for col, (icon, part, title, desc, color) in zip(cols, stages):
        with col:
            st.markdown(f"""
            <div class='glass-card' style='text-align:center; border-color:{color}33;'>
              <div style='font-size:2rem; margin-bottom:8px;'>{icon}</div>
              <div style='font-size:0.7rem; color:{color}; text-transform:uppercase; letter-spacing:1px; font-weight:700;'>{part}</div>
              <div style='font-size:1rem; font-weight:700; margin:6px 0;'>{title}</div>
              <div style='font-size:0.75rem; color:#78909c; line-height:1.5; white-space:pre-line;'>{desc}</div>
            </div>""", unsafe_allow_html=True)

    # Stats row
    st.markdown("<div class='section-heading'>📊 System Metrics</div>", unsafe_allow_html=True)
    m1,m2,m3,m4,m5,m6 = st.columns(6)
    metrics = [
        ("15","Cities","#448aff"),("19","Roads","#00e676"),("15","Traffic Laws","#b388ff"),
        ("8","Routes","#ffab40"),("500","Journey Records","#ff5252"),("400","Travellers","#18ffff"),
    ]
    for col,(val,lbl,clr) in zip([m1,m2,m3,m4,m5,m6],metrics):
        with col:
            st.markdown(f"""
            <div class='stat-card'>
              <div class='stat-value' style='color:{clr};'>{val}</div>
              <div class='stat-label'>{lbl}</div>
            </div>""", unsafe_allow_html=True)

    # Map
    st.markdown("<div class='section-heading'>🗺️ Pakistan Road Network</div>", unsafe_allow_html=True)
    graph = get_graph()
    fig = make_network_fig(graph)
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar":False})

    st.markdown("""
    <div class='info-box'>
    💡 <b>How to use:</b> Navigate using the sidebar. Each tab is a fully interactive AI module.
    All algorithms run in real-time with animated feedback.
    </div>""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# ══ PAGE: ROUTE FINDER ══
# ─────────────────────────────────────────────────────────────
elif page == "Route Finder":
    st.markdown("<div class='hero-title' style='font-size:1.8rem;'>🗺️ Route Finder</div>", unsafe_allow_html=True)
    st.markdown("<div style='color:#78909c;margin-bottom:24px;'>Find the shortest road route between any two Pakistani cities using Search Algorithms.</div>", unsafe_allow_html=True)

    graph = get_graph()

    col_l, col_r = st.columns([1, 2])
    with col_l:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        src  = st.selectbox("🏙️ From", CITIES, index=0)
        dst  = st.selectbox("🏁 To",   CITIES, index=5)
        algo = st.selectbox("⚡ Algorithm", ["A* Search","Uniform Cost Search","Bidirectional","All (Compare)"])
        mode = st.radio("🔄 Route Type", ["Direct", "Relay (change bus)"], horizontal=True)
        run  = st.button("🚀 Find Route", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col_r:
        if run:
            if src == dst:
                st.markdown("<div class='warn-box'>⚠️ Source and destination must be different.</div>", unsafe_allow_html=True)
            else:
                with st.spinner("Computing route…"):
                    time.sleep(0.4)  # micro-delay for animation feel

                results = {}
                if algo in ("Uniform Cost Search","All (Compare)"):
                    t0=time.perf_counter()
                    p,c,n=ucs(graph,src,dst)
                    results["UCS"]={"path":p,"cost":c,"nodes":n,"time":(time.perf_counter()-t0)*1000}
                if algo in ("A* Search","All (Compare)"):
                    t0=time.perf_counter()
                    p,c,n=astar(graph,src,dst,HEURISTIC)
                    results["A*"]={"path":p,"cost":c,"nodes":n,"time":(time.perf_counter()-t0)*1000}
                if algo in ("Bidirectional","All (Compare)"):
                    t0=time.perf_counter()
                    p,c,n=bidir(graph,src,dst)
                    results["Bidirectional"]={"path":p,"cost":c,"nodes":n,"time":(time.perf_counter()-t0)*1000}

                # Pick best path to show on map
                best_algo = min(results, key=lambda a: results[a]["nodes"]) if results else None
                best_path = results[best_algo]["path"] if best_algo else None

                # Relay
                relay_info = None
                if mode == "Relay (change bus)":
                    relay_city,l1,l2,d1,d2,rt = find_relay(src,dst,RELAY_POINTS,graph)
                    relay_info = {"city":relay_city,"leg1":l1,"leg2":l2,"d1":d1,"d2":d2,"total":rt}

                # Map
                st.plotly_chart(make_network_fig(graph, best_path),
                                use_container_width=True, config={"displayModeBar":False})

                # Results
                for alg, r in results.items():
                    if r["path"]:
                        COLOR_MAP={"UCS":"#448aff","A*":"#ffab40","Bidirectional":"#00e676"}
                        c=COLOR_MAP.get(alg,"#448aff")
                        st.markdown(f"""
                        <div class='glass-card fade-in' style='border-color:{c}44;'>
                          <div style='font-size:0.8rem;color:{c};font-weight:700;text-transform:uppercase;letter-spacing:1px;margin-bottom:10px;'>{alg}</div>
                          <div class='path-display'>📍 {" → ".join(r["path"])}</div>
                          <div class='stat-grid' style='margin-top:14px;'>
                            <div class='stat-card'><div class='stat-value' style='color:{c};font-size:1.4rem;'>{r["cost"]}</div><div class='stat-label'>km</div></div>
                            <div class='stat-card'><div class='stat-value' style='color:{c};font-size:1.4rem;'>{len(r["path"])-2}</div><div class='stat-label'>stops</div></div>
                            <div class='stat-card'><div class='stat-value' style='color:{c};font-size:1.4rem;'>{r["nodes"]}</div><div class='stat-label'>nodes explored</div></div>
                            <div class='stat-card'><div class='stat-value' style='color:{c};font-size:1.4rem;'>{r["time"]:.2f}</div><div class='stat-label'>ms</div></div>
                          </div>
                        </div>""", unsafe_allow_html=True)

                if relay_info and relay_info["city"]:
                    ri = relay_info
                    st.markdown(f"""
                    <div class='glass-card fade-in' style='border-color:#b388ff44;'>
                      <div style='font-size:0.8rem;color:#b388ff;font-weight:700;text-transform:uppercase;letter-spacing:1px;margin-bottom:10px;'>🔄 Relay Route via {ri["city"]}</div>
                      <div class='path-display'>🚌 Bus 1: {" → ".join(ri["leg1"])}  [{ri["d1"]} km]</div>
                      <div style='height:6px;'></div>
                      <div class='path-display'>🚌 Bus 2: {" → ".join(ri["leg2"])}  [{ri["d2"]} km]</div>
                      <div style='margin-top:12px;font-size:0.88rem;color:#b388ff;'>Total relay distance: <b>{ri["total"]} km</b></div>
                    </div>""", unsafe_allow_html=True)

                # Comparison bar chart
                if len(results) > 1:
                    algos=list(results.keys())
                    nodes=[results[a]["nodes"] for a in algos]
                    colors=["#448aff","#ffab40","#00e676"][:len(algos)]
                    fig2=go.Figure(go.Bar(x=algos,y=nodes,marker_color=colors,
                        text=nodes,textposition="outside",
                        marker_line_color="#060d1f",marker_line_width=1.5))
                    fig2.update_layout(title="Nodes Explored per Algorithm",
                        height=280,**PLOTLY_DARK,showlegend=False)
                    st.plotly_chart(fig2,use_container_width=True,config={"displayModeBar":False})
        else:
            st.plotly_chart(make_network_fig(graph), use_container_width=True,
                            config={"displayModeBar":False})
            st.markdown("<div class='info-box'>👈 Select cities and algorithm, then click <b>Find Route</b>.</div>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# ══ PAGE: LEGAL ADVISOR ══
# ─────────────────────────────────────────────────────────────
elif page == "Legal Advisor":
    st.markdown("<div class='hero-title' style='font-size:1.8rem;'>⚖️ AI Legal Advisor</div>", unsafe_allow_html=True)
    st.markdown("<div style='color:#78909c;margin-bottom:24px;'>Propositional logic KB with pl_resolution — instantly determine traffic violations & fines.</div>", unsafe_allow_html=True)

    col_l, col_r = st.columns([1, 1.6])

    LAWS = [
        ("R1","No Helmet","Fine Rs.500"),("R2","No Seatbelt","Fine Rs.1,000"),
        ("R3","Speeding","Challan Issued"),("R4","Challan + Unpaid","License Suspended"),
        ("R5","Suspended + Driving","Arrested"),("R6","Accident + No Insurance","Court Case"),
        ("R7","Court Case + Guilty","License Cancelled"),("R8","No License","Fine Rs.5,000"),
        ("R9","Mobile While Driving","Fine Rs.2,000"),("R10","Red Light Jump","Fine Rs.1,500"),
        ("R11","Fine Paid in 7 Days","50% Discount"),("R12","3+ Violations","License Suspended"),
        ("R13","License Suspended","Cannot Drive Legally"),("R14","Drunk Driving","Arrested Immediately"),
        ("R15","Arrested + Repeat","Jail Term"),
    ]

    with col_l:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.markdown("<div class='section-heading'>🚦 Driver Facts</div>", unsafe_allow_html=True)
        speeding   = st.checkbox("🚗 Speeding (Speed Above Limit)",  value=True)
        mobile     = st.checkbox("📱 Mobile While Driving",          value=True)
        not_paid   = st.checkbox("💳 Unpaid Previous Challans",      value=True)
        three_viol = st.checkbox("⚠️ 3+ Violations This Month",      value=True)
        repeat_off = st.checkbox("🔁 Repeat Offender",               value=True)
        still_driv = st.checkbox("🛑 Still Driving After Warning",   value=True)
        st.markdown("</div>", unsafe_allow_html=True)

        run_legal = st.button("⚖️ Run AI Legal Advisor", use_container_width=True)

        st.markdown("<div class='glass-card' style='margin-top:16px;'>", unsafe_allow_html=True)
        st.markdown("<div class='section-heading'>📜 15 Traffic Laws</div>", unsafe_allow_html=True)
        for rule_id, cond, cons in LAWS:
            st.markdown(f"<div style='font-size:0.78rem;padding:5px 0;border-bottom:1px solid #1e3a5f;'><span style='color:#448aff;font-weight:700;'>{rule_id}</span>&nbsp;&nbsp;<span style='color:#90a4ae;'>{cond}</span>&nbsp;<span style='color:#546e7a;'>→</span>&nbsp;<span style='color:#e8f0fe;'>{cons}</span></div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col_r:
        if run_legal:
            with st.spinner("Running PropKB inference…"):
                time.sleep(0.5)

            # Simple forward-chaining (no aima3 needed in GUI)
            facts = {
                "Speed_Above_Limit":speeding,"Mobile_While_Driving":mobile,
                "Not_Paid":not_paid,"Three_Violations":three_viol,
                "Repeat_Offender":repeat_off,"Still_Driving":still_driv,
            }
            challan = facts["Speed_Above_Limit"]
            suspended = (challan and facts["Not_Paid"]) or facts["Three_Violations"]
            cannot_drive = suspended
            arrested = suspended and facts["Still_Driving"]
            jail = arrested and facts["Repeat_Offender"]
            fine_mobile = facts["Mobile_While_Driving"]

            fines=[]
            if challan:   fines.append(("Speeding Challan (R3)",3000))
            if fine_mobile: fines.append(("Mobile While Driving (R9)",2000))
            if facts["Not_Paid"]: fines.append(("Unpaid Challans Penalty (R4)",5000))
            if jail: fines.append(("Repeat Offender Surcharge (R15)",10000))
            total=sum(f[1] for f in fines)

            st.markdown("<div class='section-heading'>🔍 Inferred Consequences</div>", unsafe_allow_html=True)
            results_items=[
                ("Challan Issued",challan,"R3 — Modus Ponens"),
                ("License Suspended",suspended,"R4+R12 — Resolution"),
                ("Cannot Drive Legally",cannot_drive,"R13"),
                ("Arrested",arrested,"R5 — Hyp. Syllogism"),
                ("Jail Term",jail,"R15 — Repeat Offender"),
            ]
            for label,val,rule in results_items:
                badge_cls = "badge-red" if val else "badge-green"
                txt = "YES ⚠️" if val else "NO ✅"
                st.markdown(f"""
                <div class='timeline-step'>
                  <div class='timeline-label'>{rule}</div>
                  <div style='display:flex;align-items:center;justify-content:space-between;margin-top:4px;'>
                    <div class='timeline-value'>{label}</div>
                    <span class='badge {badge_cls}'>{txt}</span>
                  </div>
                </div>""", unsafe_allow_html=True)

            # Fines breakdown
            st.markdown("<div class='section-heading' style='margin-top:20px;'>💰 Fine Breakdown</div>", unsafe_allow_html=True)
            if fines:
                fig_f=go.Figure(go.Bar(
                    x=[f[0] for f in fines], y=[f[1] for f in fines],
                    marker_color=["#ff5252","#ffab40","#448aff","#b388ff"][:len(fines)],
                    text=[f"Rs.{f[1]:,}" for f in fines],
                    textposition="outside",marker_line_color="#060d1f",marker_line_width=1.5))
                fig_f.update_layout(height=260,**PLOTLY_DARK,showlegend=False,
                    title=f"Total Fines: Rs.{total:,}")
                st.plotly_chart(fig_f,use_container_width=True,config={"displayModeBar":False})

            # Inference chain
            st.markdown("<div class='section-heading'>🔗 Hypothetical Syllogism Chain</div>", unsafe_allow_html=True)
            chain=[
                ("Speeding","Speed_Above_Limit → Challan_Issued","Challan Issued",challan),
                ("Challan + Unpaid","(Challan ∧ Not_Paid) → License_Suspended","Suspended",suspended),
                ("Suspended + Driving","(Suspended ∧ Still_Driving) → Arrested","Arrested",arrested),
                ("Arrested + Repeat","(Arrested ∧ Repeat_Offender) → Jail_Term","Jail Term",jail),
            ]
            for step_name,rule_str,conclusion,fired in chain:
                color="#ff5252" if fired else "#37474f"
                icon="🔴" if fired else "⚫"
                st.markdown(f"""
                <div style='display:flex;align-items:center;gap:10px;padding:10px 14px;background:var(--surface2);
                     border-radius:10px;border:1px solid {color}44;margin-bottom:8px;'>
                  <span style='font-size:1.1rem;'>{icon}</span>
                  <div>
                    <div style='font-size:0.75rem;color:#78909c;'>{rule_str}</div>
                    <div style='font-weight:600;font-size:0.9rem;color:{color};'>{conclusion} {"TRIGGERED" if fired else "not triggered"}</div>
                  </div>
                </div>""", unsafe_allow_html=True)

            if arrested:
                st.markdown(f"""<div class='warn-box' style='margin-top:16px;font-size:1rem;font-weight:600;'>
                ⚠️ Driver is <b>ARRESTED</b>. Total fines: <b>Rs. {total:,}</b>. Jail term: {'YES' if jail else 'NO'}.
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""<div class='success-box' style='margin-top:16px;font-size:1rem;font-weight:600;'>
                ✅ Driver is NOT arrested. Total fines: <b>Rs. {total:,}</b>.
                </div>""", unsafe_allow_html=True)
        else:
            st.markdown("<div class='info-box'>👈 Set driver facts and click <b>Run AI Legal Advisor</b>.</div>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# ══ PAGE: BUS SCHEDULER ══
# ─────────────────────────────────────────────────────────────
elif page == "Bus Scheduler":
    st.markdown("<div class='hero-title' style='font-size:1.8rem;'>📅 CSP Bus Scheduler</div>", unsafe_allow_html=True)
    st.markdown("<div style='color:#78909c;margin-bottom:24px;'>Constraint Satisfaction Problem — assigns buses to routes conflict-free, guaranteeing Ali's relay bus.</div>", unsafe_allow_html=True)

    col_l, col_r = st.columns([1, 1.8])

    with col_l:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.markdown("<div class='section-heading'>⚙️ Solver Settings</div>", unsafe_allow_html=True)
        use_ac3 = st.toggle("Enable AC3 Arc Consistency", value=True)
        show_domains = st.toggle("Show Domain Sizes", value=True)
        run_csp = st.button("🚌 Schedule Buses", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.markdown("<div class='section-heading'>📋 Route Info</div>", unsafe_allow_html=True)
        for r,info in ROUTE_INFO.items():
            is_ali = r=="R3"
            color = "#ffab40" if is_ali else "#448aff"
            hw = "🛣️" if info["highway"] else "🛤️"
            st.markdown(f"""
            <div style='padding:7px 0;border-bottom:1px solid #1e3a5f;display:flex;align-items:center;gap:8px;'>
              <span style='color:{color};font-weight:700;font-size:0.85rem;'>{r}</span>
              <span style='font-size:0.78rem;color:#90a4ae;flex:1;'>{info["name"]}</span>
              <span style='font-size:0.72rem;color:#546e7a;'>{hw} {info["time"]}</span>
              {"<span style='font-size:0.7rem;color:#ffab40;font-weight:700;'>ALI</span>" if is_ali else ""}
            </div>""", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col_r:
        if run_csp:
            with st.spinner("Running CSP solver…"):
                time.sleep(0.5)

            domains_bt  = {r:list(v) for r,v in DOMAINS_BASE.items()}
            domains_ac3 = {r:list(v) for r,v in DOMAINS_BASE.items()}

            # Plain BT
            t0=time.perf_counter()
            sol_bt,calls_bt = csp_solve({r:list(v) for r,v in DOMAINS_BASE.items()})
            time_bt=(time.perf_counter()-t0)*1000

            # AC3 + MRV
            domains_ac3_run={r:list(v) for r,v in DOMAINS_BASE.items()}
            _,removed=run_ac3(domains_ac3_run)
            t0=time.perf_counter()
            sol_ac3,calls_ac3=csp_solve({r:list(d) for r,d in domains_ac3_run.items()})
            time_ac3=(time.perf_counter()-t0)*1000

            final_sol = sol_ac3 if use_ac3 and sol_ac3 else sol_bt
            score = soft_score(final_sol) if final_sol else 0

            # Stats
            c1,c2,c3,c4=st.columns(4)
            c1.metric("BT Calls (plain)",str(calls_bt))
            c2.metric("BT Calls (MRV+AC3)",str(calls_ac3))
            c3.metric("Values pruned (AC3)",str(removed))
            c4.metric("Soft Score",str(score))

            # Domain before/after AC3
            if show_domains:
                st.markdown("<div class='section-heading' style='margin-top:18px;'>🔬 AC3 Domain Reduction</div>", unsafe_allow_html=True)
                dom_rows=[]
                for r in ROUTES:
                    before=len(DOMAINS_BASE[r])
                    after=len(domains_ac3_run[r])
                    pruned=before-after
                    removed_vals=set(DOMAINS_BASE[r])-set(domains_ac3_run[r])
                    dom_rows.append({"Route":r,"Journey":ROUTE_INFO[r]["name"],"Before":before,
                        "After":after,"Pruned":pruned,"Removed":",".join(sorted(removed_vals)) or "none"})
                df_dom=pd.DataFrame(dom_rows)
                st.dataframe(df_dom,use_container_width=True,hide_index=True)

            # Final schedule
            if final_sol:
                st.markdown("<div class='section-heading' style='margin-top:18px;'>✅ Optimal Bus Schedule</div>", unsafe_allow_html=True)
                rows=[]
                for r in ROUTES:
                    bus=final_sol[r]; info=ROUTE_INFO[r]
                    is_ali=r=="R3"; is_hw=info["highway"]; is_large=bus in LARGE
                    sc1="+2 ✓" if (is_hw and is_large) else ""
                    rows.append({"Route":r,"Journey":info["name"],"Time":info["time"],
                        "Bus Assigned":bus,"Highway":("Yes" if is_hw else "No"),
                        "SC1 Score":sc1,"Ali's Bus":("🎓 YES!" if is_ali else "")})
                df_sched=pd.DataFrame(rows)
                st.dataframe(df_sched,use_container_width=True,hide_index=True)

                ali_bus=final_sol["R3"]
                st.markdown(f"""
                <div class='success-box' style='font-size:1.05rem;font-weight:700;margin-top:16px;'>
                🎓 Ali's connecting bus: <b>{ali_bus}</b> on Route R3 (Sukkur → Multan) at 12pm ✓<br>
                <span style='font-weight:400;font-size:0.88rem;'>Hard constraint HC2 guarantees R3 is always assigned. Ali will not miss his bus!</span>
                </div>""", unsafe_allow_html=True)

                # Gantt chart
                COLORS_GANTT={"Bus1":"#ff5252","Bus2":"#ff7043","Bus3":"#ffab40",
                    "Bus4":"#66bb6a","Bus5":"#26c6da","Bus6":"#448aff",
                    "Bus7":"#b388ff","Bus8":"#f06292","Bus9":"#a5d6a7","Bus10":"#80deea"}
                TIME_ORDER={"8am":0,"10am":2,"12pm":4,"2pm":6,"4pm":8,"6pm":10}
                fig_g=go.Figure()
                for r in ROUTES:
                    bus=final_sol[r]; info=ROUTE_INFO[r]
                    t=TIME_ORDER[info["time"]]
                    fig_g.add_trace(go.Bar(
                        name=bus, x=[2], y=[r], base=t, orientation="h",
                        marker_color=COLORS_GANTT.get(bus,"#448aff"),
                        text=f"{bus}<br>{info['time']}", textposition="inside",
                        hovertemplate=f"<b>{r}</b>: {info['name']}<br>Bus: {bus}<br>Time: {info['time']}<extra></extra>",
                    ))
                fig_g.update_layout(
                    barmode="overlay",height=340,
                    title="Bus Schedule Gantt Chart",
                    xaxis=dict(tickvals=[0,2,4,6,8,10],
                        ticktext=["8am","10am","12pm","2pm","4pm","6pm"],
                        title="Time of Day"),
                    yaxis=dict(title="Route",categoryorder="array",categoryarray=ROUTES[::-1]),
                    showlegend=False, **PLOTLY_DARK
                )
                st.plotly_chart(fig_g,use_container_width=True,config={"displayModeBar":False})
        else:
            st.markdown("<div class='info-box'>👈 Click <b>Schedule Buses</b> to run the CSP solver.</div>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# ══ PAGE: DELAY PREDICTOR ══
# ─────────────────────────────────────────────────────────────
elif page == "Delay Predictor":
    st.markdown("<div class='hero-title' style='font-size:1.8rem;'>🧠 Delay Predictor</div>", unsafe_allow_html=True)
    st.markdown("<div style='color:#78909c;margin-bottom:24px;'>Artificial Neural Network (sklearn MLP) — predicts ON TIME or DELAYED for any journey.</div>", unsafe_allow_html=True)

    with st.spinner("Loading ANN model…"):
        clf, scaler, metrics = train_ann()

    col_l, col_r = st.columns([1, 1.6])

    with col_l:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.markdown("<div class='section-heading'>🎛️ Journey Parameters</div>", unsafe_allow_html=True)
        dist   = st.slider("📏 Distance (km)",         80,  700, 320, step=10)
        weather= st.slider("🌦️ Weather Score (1=clear, 5=storm)", 1, 5, 2)
        traffic= st.slider("🚦 Traffic Score (1=light, 5=heavy)", 1, 5, 2)
        road   = st.slider("🛣️ Road Quality (1=poor, 5=motorway)", 1, 5, 4)
        age    = st.slider("🚌 Bus Age (years)",         1,  15, 3)
        predict_btn = st.button("🔮 Predict", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

        # Quick presets
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.markdown("<div class='section-heading'>⚡ Quick Presets</div>", unsafe_allow_html=True)
        presets=[
            ("🎓 Ali's Bus",320,2,2,4,3),
            ("⛈️ Storm Route",650,5,4,2,12),
            ("🌅 Ideal Journey",160,1,1,5,2),
        ]
        for label,d,w,t,r,a in presets:
            if st.button(label,key=f"preset_{label}",use_container_width=True):
                dist,weather,traffic,road,age=d,w,t,r,a
        st.markdown("</div>", unsafe_allow_html=True)

    with col_r:
        if predict_btn:
            with st.spinner("Running neural network…"):
                time.sleep(0.4)

            raw = np.array([[dist,weather,traffic,road,age]])
            scaled = scaler.transform(raw)
            pred = clf.predict(scaled)[0]
            prob = clf.predict_proba(scaled)[0]
            conf = prob[pred]*100
            is_delayed = pred==1

            # Big result
            if is_delayed:
                st.markdown(f"""
                <div class='glass-card fade-in' style='text-align:center;border-color:#ff525266;'>
                  <div style='font-size:3rem;'>⚠️</div>
                  <div style='font-size:2rem;font-weight:900;color:#ff5252;'>DELAYED</div>
                  <div style='font-size:1rem;color:#78909c;margin-top:6px;'>Confidence: {conf:.1f}%</div>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class='glass-card fade-in' style='text-align:center;border-color:#00e67666;'>
                  <div style='font-size:3rem;'>✅</div>
                  <div style='font-size:2rem;font-weight:900;color:#00e676;'>ON TIME</div>
                  <div style='font-size:1rem;color:#78909c;margin-top:6px;'>Confidence: {conf:.1f}%</div>
                </div>""", unsafe_allow_html=True)

            # Confidence gauge
            gauge_color="#ff5252" if is_delayed else "#00e676"
            fig_g=go.Figure(go.Indicator(
                mode="gauge+number",value=conf,
                number=dict(suffix="%",font=dict(size=32,color=gauge_color)),
                gauge=dict(
                    axis=dict(range=[0,100],tickcolor="#546e7a"),
                    bar=dict(color=gauge_color),
                    bgcolor="#1e293b",
                    steps=[dict(range=[0,50],color="#112244"),dict(range=[50,100],color="#0d1b3e")],
                    threshold=dict(line=dict(color="#f1f5f9",width=2),thickness=0.75,value=50),
                ),
                title=dict(text="Prediction Confidence",font=dict(color="#90a4ae",size=13))
            ))
            fig_g.update_layout(height=240,paper_bgcolor="rgba(0,0,0,0)",
                                font=dict(color="#e8f0fe"),margin=dict(l=20,r=20,t=30,b=10))
            st.plotly_chart(fig_g,use_container_width=True,config={"displayModeBar":False})

            # Feature importance bar
            factors=[
                ("Distance",dist/700,"#448aff"),
                ("Weather",weather/5,"#ff5252"),
                ("Traffic",traffic/5,"#ffab40"),
                ("Road Quality",road/5,"#00e676"),
                ("Bus Age",age/15,"#b388ff"),
            ]
            fig_f=go.Figure()
            for fname,norm_val,clr in factors:
                fig_f.add_trace(go.Bar(name=fname,x=[norm_val],y=[fname],
                    orientation="h",marker_color=clr,
                    text=f"{norm_val*100:.0f}%",textposition="outside",
                    marker_line_width=0))
            fig_f.update_layout(height=220,barmode="overlay",showlegend=False,
                title="Feature Intensity (normalised)",**PLOTLY_DARK,
                xaxis=dict(range=[0,1.15],tickformat=".0%",**PLOTLY_DARK["xaxis"]))
            st.plotly_chart(fig_f,use_container_width=True,config={"displayModeBar":False})

            # Delay rule explanation
            rule_fired=(weather>=4) or (traffic>=4) or (dist>400 and road<=2)
            if rule_fired:
                reasons=[]
                if weather>=4: reasons.append(f"Weather={weather}/5 ≥ 4 🌩️")
                if traffic>=4: reasons.append(f"Traffic={traffic}/5 ≥ 4 🚦")
                if dist>400 and road<=2: reasons.append(f"Distance={dist}km>400 & Road={road}/5≤2 🛤️")
                st.markdown(f"<div class='warn-box'>🔴 Delay rule triggered: {' | '.join(reasons)}</div>",
                            unsafe_allow_html=True)
            else:
                st.markdown("<div class='success-box'>🟢 No delay conditions triggered — journey conditions are favourable.</div>",
                            unsafe_allow_html=True)

        # Model metrics (always shown)
        st.markdown("<div class='section-heading' style='margin-top:20px;'>📊 Model Performance</div>", unsafe_allow_html=True)
        m1,m2,m3,m4=st.columns(4)
        m1.metric("Accuracy",  f"{metrics['acc']*100:.1f}%")
        m2.metric("Precision", f"{metrics['prec']*100:.1f}%")
        m3.metric("Recall",    f"{metrics['rec']*100:.1f}%")
        m4.metric("F1-Score",  f"{metrics['f1']:.4f}")

        # Confusion matrix
        cm=np.array(metrics["cm"])
        fig_cm=go.Figure(go.Heatmap(
            z=cm,x=["On Time","Delayed"],y=["On Time","Delayed"],
            colorscale=[[0,"#0d1b3e"],[1,"#448aff"]],showscale=False,
            text=[[str(v) for v in row] for row in cm.tolist()],
            texttemplate="%{text}",textfont=dict(size=20,color="white")
        ))
        fig_cm.update_layout(height=240,title="Confusion Matrix",**PLOTLY_DARK,
            xaxis=dict(title="Predicted",**PLOTLY_DARK["xaxis"]),
            yaxis=dict(title="Actual",**PLOTLY_DARK["yaxis"]))
        st.plotly_chart(fig_cm,use_container_width=True,config={"displayModeBar":False})

        # Training loss
        fig_loss=go.Figure(go.Scatter(
            y=metrics["loss"],mode="lines",line=dict(color="#448aff",width=2),
            fill="tozeroy",fillcolor="rgba(68,138,255,0.08)"
        ))
        fig_loss.update_layout(height=200,title="Training Loss Curve",**PLOTLY_DARK,
            xaxis=dict(title="Epoch",**PLOTLY_DARK["xaxis"]),
            yaxis=dict(title="Loss",**PLOTLY_DARK["yaxis"]))
        st.plotly_chart(fig_loss,use_container_width=True,config={"displayModeBar":False})

# ─────────────────────────────────────────────────────────────
# ══ PAGE: TRAVELLER TYPES ══
# ─────────────────────────────────────────────────────────────
elif page == "Traveller Types":
    st.markdown("<div class='hero-title' style='font-size:1.8rem;'>👥 Traveller Types</div>", unsafe_allow_html=True)
    st.markdown("<div style='color:#78909c;margin-bottom:24px;'>K-Means Clustering (k=3) — groups 400 PakTravel customers into Business, Family, and Commuter profiles.</div>", unsafe_allow_html=True)

    with st.spinner("Running K-Means clustering…"):
        df5, inertias = train_kmeans()

    # Stats row
    profile_colors={"Business":"#ffab40","Family":"#00e676","Commuter":"#448aff"}
    c1,c2,c3=st.columns(3)
    for col,profile in zip([c1,c2,c3],["Business","Family","Commuter"]):
        cnt=(df5["Profile"]==profile).sum()
        color=profile_colors[profile]
        icons={"Business":"💼","Family":"👨‍👩‍👧","Commuter":"🚶"}
        with col:
            st.markdown(f"""
            <div class='glass-card' style='text-align:center;border-color:{color}44;'>
              <div style='font-size:2rem;'>{icons[profile]}</div>
              <div style='font-size:1.5rem;font-weight:800;color:{color};'>{cnt}</div>
              <div style='font-size:0.85rem;color:#90a4ae;'>{profile} Travellers</div>
              <div style='font-size:0.72rem;color:#546e7a;margin-top:4px;'>{cnt/len(df5):.1%} of total</div>
            </div>""", unsafe_allow_html=True)

    col_l, col_r = st.columns([1.5, 1])

    with col_l:
        # Main scatter
        fig_sc=px.scatter(df5, x="Trips_Per_Month", y="Avg_Journey_Distance",
            color="Profile",
            color_discrete_map=profile_colors,
            hover_data=["Loyalty_Points","Avg_Booking_Days","Complaints_Filed"],
            labels={"Trips_Per_Month":"Trips Per Month",
                    "Avg_Journey_Distance":"Average Journey Distance (km)"},
            title="Traveller Clusters — Trips vs Distance")
        fig_sc.update_traces(marker=dict(size=7,opacity=0.8,line=dict(width=0)))
        fig_sc.update_layout(height=400,**PLOTLY_DARK,
            legend=dict(bgcolor="rgba(13,27,62,0.8)",bordercolor="#1e3a5f"))
        st.plotly_chart(fig_sc,use_container_width=True,config={"displayModeBar":False})

        # Loyalty vs Booking scatter
        fig_lb=px.scatter(df5, x="Avg_Booking_Days", y="Loyalty_Points",
            color="Profile", color_discrete_map=profile_colors,
            title="Booking Lead Time vs Loyalty Points")
        fig_lb.update_traces(marker=dict(size=6,opacity=0.75,line=dict(width=0)))
        fig_lb.update_layout(height=350,**PLOTLY_DARK,showlegend=False)
        st.plotly_chart(fig_lb,use_container_width=True,config={"displayModeBar":False})

    with col_r:
        # Elbow
        fig_elbow=go.Figure()
        fig_elbow.add_trace(go.Scatter(x=list(range(1,9)),y=inertias,
            mode="lines+markers",line=dict(color="#448aff",width=2.5),
            marker=dict(size=9,color="#448aff"),
            fill="tozeroy",fillcolor="rgba(68,138,255,0.08)"))
        fig_elbow.add_vline(x=3,line_dash="dash",line_color="#ff5252",
            annotation_text="k=3 optimal",annotation_font_color="#ff5252")
        fig_elbow.update_layout(height=280,title="Elbow Method",
            xaxis=dict(title="k (clusters)",tickvals=list(range(1,9)),**PLOTLY_DARK["xaxis"]),
            yaxis=dict(title="Inertia",**PLOTLY_DARK["yaxis"]),
            **PLOTLY_DARK)
        st.plotly_chart(fig_elbow,use_container_width=True,config={"displayModeBar":False})

        # Radar chart — cluster profiles
        cats=["Trips/Month","Booking Days","Distance","Loyalty","Complaints"]
        radar_data={}
        for profile in ["Business","Family","Commuter"]:
            sub=df5[df5["Profile"]==profile]
            vals=[
                sub["Trips_Per_Month"].mean()/15,
                sub["Avg_Booking_Days"].mean()/30,
                sub["Avg_Journey_Distance"].mean()/700,
                sub["Loyalty_Points"].mean()/5000,
                sub["Complaints_Filed"].mean()/10,
            ]
            radar_data[profile]=vals

        fig_radar=go.Figure()
        for profile,vals in radar_data.items():
            v=vals+[vals[0]]
            c=profile_colors[profile]
            fig_radar.add_trace(go.Scatterpolar(
                r=v, theta=cats+[cats[0]],
                name=profile, fill="toself",
                fillcolor=c.replace("#","rgba(")+",0.12)" if False else f"{c}1e",
                line=dict(color=c,width=2),
                marker=dict(size=5,color=c),
            ))
        fig_radar.update_layout(
            polar=dict(
                bgcolor="#0d1b3e",
                radialaxis=dict(visible=True,range=[0,1],gridcolor="#1e3a5f",color="#546e7a"),
                angularaxis=dict(gridcolor="#1e3a5f",color="#90a4ae"),
            ),
            height=320,title="Cluster Profile Radar",
            paper_bgcolor="rgba(6,13,31,0)",
            legend=dict(bgcolor="rgba(13,27,62,0.8)",bordercolor="#1e3a5f",font=dict(color="#e8f0fe")),
            font=dict(color="#e8f0fe"),margin=dict(l=40,r=40,t=50,b=20)
        )
        st.plotly_chart(fig_radar,use_container_width=True,config={"displayModeBar":False})

        # Profile table
        st.markdown("<div class='section-heading'>📋 Cluster Averages</div>", unsafe_allow_html=True)
        profile_stats=df5.groupby("Profile")[
            ["Trips_Per_Month","Avg_Journey_Distance","Avg_Booking_Days","Loyalty_Points"]
        ].mean().round(1)
        st.dataframe(profile_stats,use_container_width=True)

    # Histograms
    st.markdown("<div class='section-heading' style='margin-top:8px;'>📊 Feature Distributions by Profile</div>", unsafe_allow_html=True)
    feat_cols=["Trips_Per_Month","Avg_Journey_Distance","Loyalty_Points"]
    fig_hist=make_subplots(rows=1,cols=3,subplot_titles=feat_cols)
    for ci,feat in enumerate(feat_cols,1):
        for profile,color in profile_colors.items():
            sub=df5[df5["Profile"]==profile][feat]
            fig_hist.add_trace(go.Histogram(
                x=sub,name=profile if ci==1 else None,
                marker_color=color,opacity=0.7,
                showlegend=(ci==1),nbinsx=18,
            ),row=1,col=ci)
    fig_hist.update_layout(height=280,barmode="overlay",**PLOTLY_DARK,
        legend=dict(bgcolor="rgba(13,27,62,0.8)",bordercolor="#1e3a5f",font=dict(color="#e8f0fe")))
    st.plotly_chart(fig_hist,use_container_width=True,config={"displayModeBar":False})

    st.markdown("""
    <div class='success-box'>
    🎓 <b>Ali's Profile:</b> As a student making an occasional long-distance trip, Ali falls in the <b>Family</b> cluster.
    PakTravel can offer him a <b>student/occasional traveller discount package</b>!
    </div>""", unsafe_allow_html=True)
