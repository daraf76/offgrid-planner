import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

# --- CONFIGURAZIONE ---
st.set_page_config(page_title="Off-Grid Planner V15", layout="wide", page_icon="☀️")

# --- CSS (FIX LEGGIBILITÀ) ---
st.markdown("""
<style>
    /* Tab Styling */
    .stTabs [data-baseweb="tab-list"] { border-bottom: 1px solid #444; }
    .stTabs [data-baseweb="tab"] { background-color: #262730; color: #fff; border: 1px solid #444; }
    .stTabs [aria-selected="true"] { background-color: #0e1117; border-top: 3px solid #FFD700; color: #fff; font-weight: bold; }
    
    /* Metrics Box Styling */
    [data-testid="stMetric"] { 
        background-color: #ffffff; 
        border: 1px solid #ddd; 
        padding: 15px; 
        border-radius: 10px; 
        color: #000000;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
    }
    [data-testid="stMetricValue"] { 
        color: #000000 !important; 
        font-size: 24px; 
        font-weight: bold;
    }
    [data-testid="stMetricLabel"] { 
        color: #333333 !important; 
        font-size: 14px;
        font-weight: 500;
    }
    
    /* Summary Box: Dark Mode */
    .summary-box { 
        background-color: #000000; 
        color: #ffffff; 
        padding: 20px; 
        border-radius: 12px; 
        border: 1px solid #333;
        border-left: 6px solid #FFD700; 
        margin-bottom: 15px; 
        font-family: 'Courier New', monospace;
    }
    .summary-box b { color: #FFD700; }
    .summary-box hr { border-color: #333; }
    
    .icon-header { font-size: 18px; font-weight: bold; margin-bottom: 10px; border-bottom: 2px solid #FFD700; display: inline-block; }
</style>
""", unsafe_allow_html=True)

st.title("☀️ Off-Grid Planner V15 by RD")

# --- ALGORITMO SMART LAYOUT ---
def get_grid_fit(space_w, space_h, item_w, item_h, gap):
    if space_w < item_w or space_h < item_h: return 0, 0, 0, 0, 0
    cols = int((space_w + gap) / (item_w + gap))
    rows = int((space_h + gap) / (item_h + gap))
    used_w = cols * item_w + (cols - 1) * gap if cols > 0 else 0
    used_h = rows * item_h + (rows - 1) * gap if rows > 0 else 0
    return cols, rows, cols*rows, used_w, used_h

def calculate_smart_layout(roof_w, roof_h, pan_w, pan_h, margin, gap):
    uw = roof_w - (2 * margin); uh = roof_h - (2 * margin)
    if uw <= 0 or uh <= 0: return 0, [], "Spazio Insufficiente"

    strategies_results = []
    scenarios = [
        ("Solo Verticale", pan_w, pan_h, None, None),
        ("Solo Orizzontale", pan_h, pan_w, None, None),
        ("Misto (V+O)", pan_w, pan_h, pan_h, pan_w),
        ("Misto (O+V)", pan_h, pan_w, pan_w, pan_h),
    ]

    for name, mw, mh, fw, fh in scenarios:
        panels = []
        mc, mr, m_count, m_used_w, m_used_h = get_grid_fit(uw, uh, mw, mh, gap)
        start_x_m = margin; start_y_m = margin
        for c in range(mc):
            for r in range(mr):
                panels.append((start_x_m + c*(mw+gap), start_y_m + r*(mh+gap), mw, mh))
        total_count = m_count
        
        if fw and fh:
            rem_w_r = uw - m_used_w - gap; rem_h_r = uh
            rc, rr, r_count, _, _ = get_grid_fit(rem_w_r, rem_h_r, fw, fh, gap)
            rem_w_b = uw; rem_h_b = uh - m_used_h - gap
            bc, br, b_count, _, _ = get_grid_fit(rem_w_b, rem_h_b, fw, fh, gap)
            
            if r_count >= b_count and r_count > 0:
                sx = margin + m_used_w + gap; sy = margin
                for c in range(rc):
                    for r in range(rr):
                        panels.append((sx + c*(fw+gap), sy + r*(fh+gap), fw, fh))
                total_count += r_count
            elif b_count > r_count and b_count > 0:
                sx = margin; sy = margin + m_used_h + gap
                for c in range(bc):
                    for r in range(br):
                        panels.append((sx + c*(fw+gap), sy + r*(fh+gap), fw, fh))
                total_count += b_count
        strategies_results.append((total_count, panels, name))

    strategies_results.sort(key=lambda x: x[0], reverse=True)
    return strategies_results[0]

def plot_layout(roof_w, roof_h, panels, title):
    fig, ax = plt.subplots(figsize=(6, 4))
    fig.patch.set_facecolor('white'); ax.set_facecolor('#e6e6e6')
    ax.add_patch(patches.Rectangle((0, 0), roof_w, roof_h, linewidth=3, edgecolor='#333', facecolor='none'))
    for (x, y, w, h) in panels:
        color = '#0d47a1' if h > w else '#1976d2'
        ax.add_patch(patches.Rectangle((x, y), w, h, linewidth=1, edgecolor='#eee', facecolor=color, alpha=0.9))
    ax.set_xlim(-0.5, roof_w + 0.5); ax.set_ylim(roof_h + 0.5, -0.5); ax.set_aspect('equal')
    ax.set_title(title); ax.axis('on')
    return fig

# --- INIT VARIABLES ---
fabbisogno_base_kwh=0; kwh_ev=0; kwh_pdc_inv=0; kw_pv_tot=0; kwh_batt_tot=0; totale_euro=0
totale_pannelli_progetto=0; active_falde_report=[]
kw_picco_list=[]; voc_stringa_freddo=0; ampere_max=0; cavo_rec=0
prod_inv=0; prod_est=0; surplus_est=0; picco_potenza_kW=0; n_batt=0

# --- SIDEBAR ---
st.sidebar.header("📍 Dati Progetto")
client_name = st.sidebar.text_input("Nome", "Mario Rossi")
project_name = st.sidebar.text_input("Progetto", "Baita Alpina")
giorni_autonomia = st.sidebar.slider("Giorni Autonomia", 1.0, 5.0, 2.0, step=0.5)
st.sidebar.markdown("---")
ore_sole_inverno = st.sidebar.number_input("Ore Sole Inverno (PSH)", value=2.5, step=0.1)
ore_sole_estate = st.sidebar.number_input("Ore Sole Estate (PSH)", value=6.0, step=0.1)
efficienza_sistema = st.sidebar.slider("Efficienza (%)", 70, 95, 85) / 100

# --- TABS ---
# AGGIUNTO TAB INFO ALLA FINE
tab_loads, tab_design, tab_capex, tab_electric, tab_winter, tab_summer, tab_report, tab_info = st.tabs([
    "🏠 Consumi", "📐 Progettazione", "🛠️ Costi", "⚡ Param.Elettrici", "❄️ Inverno", "☀️ Estate", "📋 Riepilogo", "ℹ️ Metodo di Calcolo"
])

# --- TAB 1: CONSUMI ---
with tab_loads:
    c1, c2, c3 = st.columns(3)
    
    # Inizializziamo liste separate per gestire meglio i picchi
    loads_power = [] # Potenze nominali
    
    # 1. BASE
    with c1:
        st.markdown('<div class="icon-header">💡 Carichi Casa</div>', unsafe_allow_html=True)
        base = {
            "Frigo": {"kwh": 1.6, "w": 150}, 
            "Luci": {"kwh": 0.7, "w": 100}, 
            "TV/PC": {"kwh": 1.3, "w": 300}, 
            "Lavatrice": {"kwh": 1.2, "w": 2200}, 
            "Lavastoviglie": {"kwh": 1.3, "w": 2200}, 
            "Clima": {"kwh": 1.0, "w": 1000},
            "Pompa acqua": {"kwh": 0.9, "w": 1000},    
            "Forno": {"kwh": 1.6, "w": 2500},
            "Phon": {"kwh": 0.4, "w": 2000}
        }
        sel = st.multiselect("Dispositivi:", list(base.keys()), default=["Frigo","Luci","TV/PC","Lavatrice"])
        
        kwh_base = 0
        for item in sel:
            kwh_base += base[item]["kwh"]
            loads_power.append(base[item]["w"])
            
        fabbisogno_base_kwh = kwh_base
        st.info(f"Consumo Base: **{round(fabbisogno_base_kwh, 1)} kWh**")

    # 2. RISCALDAMENTO
    with c2:
        st.markdown('<div class="icon-header">🔥 Riscaldamento</div>', unsafe_allow_html=True)
        kw_pdc_p = 0
        if st.checkbox("Usa PdC/Boiler"):
            if st.radio("Tipo:", ["Boiler ACS", "PdC Riscaldamento"]) == "Boiler ACS":
                pot = st.number_input("kW Resistenza", 1.0, 3.0, 1.2, step=0.1)
                ore = st.slider("Ore Boiler", 1, 5, 2)
                kwh_pdc_inv = pot * ore
                loads_power.append(pot * 1000)
            else:
                mq = st.number_input("Mq Casa", 30, 200, 80)
                iso = st.select_slider("Coibentazione", ["Pessima", "Media", "Ottima"])
                cf = 0.12 if iso=="Pessima" else (0.08 if iso=="Media" else 0.05)
                kwh_pdc_inv = (mq*cf/3)*8
                # La PdC ha un motore, lo consideriamo nel picco
                loads_power.append((mq*cf/3)*1000)
        st.info(f"Consumo Termico: **{round(kwh_pdc_inv, 1)} kWh**")

    # 3. EV
    with c3:
        st.markdown('<div class="icon-header">🚗 EV</div>', unsafe_allow_html=True)
        kw_ev_p = 0
        if st.checkbox("Ricarica EV"):
            km = st.number_input("Km/gg", 10, 200, 40)
            cons = st.number_input("kWh/100km", 10.0, 30.0, 18.0)
            kwh_ev = (km/100)*cons
            wb = st.selectbox("Wallbox kW", [2.3, 3.7, 7.4])
            # La Wallbox è un carico resistivo costante e pesante
            loads_power.append(wb * 1000)
            st.metric("Ricarica", f"{round(kwh_ev, 1)} kWh")

    # --- CALCOLO INVERTER CORRETTO ---
    fabbisogno_invernale_tot = fabbisogno_base_kwh + kwh_pdc_inv + kwh_ev
    
    if loads_power:
        # 1. Somma di tutto con contemporaneità 0.7 (più sicuro di 0.6)
        scenario_contemporaneo = sum(loads_power) * 0.6
        
        # 2. Carico singolo più grande con fattore di spunto 1.5
        # (Es. una PdC da 3kW richiede spunto, una Wallbox no, ma per sicurezza stiamo larghi)
        max_load = max(loads_power)
        scenario_spunto = max_load * 1.5
        
        # 3. Scegliamo il peggiore dei due casi
        inverter_netto = max(scenario_contemporaneo, scenario_spunto)
        
        # 4. Aggiungiamo margine sicurezza hardware +20%
        picco_potenza_kW = (inverter_netto * 1.2) / 1000
    else:
        picco_potenza_kW = 3.0 # Minimo sindacale se non seleziona nulla

    st.divider()
    c_res1, c_res2 = st.columns(2)
    c_res1.metric("Fabbisogno Invernale", f"{round(fabbisogno_invernale_tot, 1)} kWh/gg")
    c_res2.metric("Inverter Consigliato", f"{round(picco_potenza_kW, 1)} kW", help="Include margine 20% e gestione spunti motori")

# --- TAB 2: DESIGN ---
with tab_design:
    st.subheader("Layout Intelligente")
    with st.expander("🛠️ Dati Pannello & Margini", expanded=True):
        ce1, ce2, ce3, ce4, ce5, ce6 = st.columns(6)
        pan_w_m = ce1.number_input("Larghezza (m)", value=1.134, format="%.3f")
        pan_h_m = ce2.number_input("Altezza (m)", value=1.722, format="%.3f")
        pan_watt = ce3.number_input("Wp", value=430)
        pan_voc = ce4.number_input("Voc", value=38.0)
        margin_m = ce5.number_input("Margine (m)", value=0.30)
        gap_m = ce6.number_input("Morsetto (m)", value=0.02)

    cf1, cf2 = st.columns(2)
    active_falde = []
    with cf1:
        if st.checkbox("Attiva F1", value=True):
            w1 = st.number_input("L1 (m)", min_value=1.0, value=6.3, step=0.1)
            h1 = st.number_input("H1 (m)", min_value=1.0, value=5.2, step=0.1)
            active_falde.append({"id":1, "w":w1, "h":h1})
    with cf2:
        if st.checkbox("Attiva F2"):
            w2 = st.number_input("L2 (m)", min_value=1.0, value=4.5, step=0.1)
            h2 = st.number_input("H2 (m)", min_value=1.0, value=3.8, step=0.1)
            active_falde.append({"id":2, "w":w2, "h":h2})

    st.divider()
    if active_falde:
        cols_vis = st.columns(len(active_falde))
        for idx, falda in enumerate(active_falde):
            count, panels, strat = calculate_smart_layout(falda['w'], falda['h'], pan_w_m, pan_h_m, margin_m, gap_m)
            totale_pannelli_progetto += count
            active_falde_report.append(f"Falda {falda['id']}: {count} Moduli ({strat})")
            with cols_vis[idx]:
                st.markdown(f"**Falda {falda['id']}**")
                if count > 0:
                    st.success(f"✅ {count} Mod. - {strat}")
                    st.pyplot(plot_layout(falda['w'], falda['h'], panels, strat))
                else: st.error("Spazio Insufficiente")
                    
    if totale_pannelli_progetto == 0: totale_pannelli_progetto = 10
    kw_pv_tot = (totale_pannelli_progetto * pan_watt) / 1000
    st.metric("TOTALE IMPIANTO", f"{round(kw_pv_tot, 2)} kWp")

# --- TAB 3: COSTI ---
with tab_capex:
    st.subheader("Analisi Economica")
    req_kwh = fabbisogno_invernale_tot * giorni_autonomia
    n_batt = int(req_kwh/5.12)+1
    col_k1, col_k2 = st.columns(2)
    with col_k1:
        n_pan = st.number_input("N. Pannelli", value=totale_pannelli_progetto)
        c_pan = st.number_input("€ Pannello", value=150)
        c_str = st.number_input("€ Struttura/mod", value=45)
        c_inv = st.number_input("€ Inverter", value=1500)
    with col_k2:
        n_b = st.number_input("N. Batterie", value=n_batt)
        c_batt = st.number_input("€ Batteria", value=1300)
        c_bos = st.number_input("€ BOS/Inst.", value=2500)
    
    totale_euro = (n_pan*(c_pan+c_str)) + c_inv + (n_b*c_batt) + c_bos
    kwh_batt_tot = n_b * 5.12
    st.metric("TOTALE STIMATO", f"€ {totale_euro:,.0f}")

# --- TAB 4: ELETTRICA ---
with tab_electric:
    st.subheader("Dimensionamento")
    ce1, ce2 = st.columns(2)
    with ce1:
        max_voc = st.number_input("Max Voc Inv.", value=450)
        n_ser = st.slider("Moduli in Serie", 1, n_pan, int(n_pan/2) if n_pan>1 else 1)
        voc_stringa_freddo = n_ser * pan_voc * 1.15
        st.metric("Voc Stringa (-10°C)", f"{int(voc_stringa_freddo)} V")
        if voc_stringa_freddo > max_voc: st.error("⚠️ ALTA TENSIONE!")
        else: st.success("✅ Tensione OK")
    with ce2:
        dist = st.number_input("Distanza m", 2.0)
        ampere_max = (picco_potenza_kW*1000)/48
        sez = (2*dist*ampere_max*0.018)/0.5
        cavo_rec = 95 if sez>70 else (70 if sez>50 else (50 if sez>35 else 35))
        st.metric("Corrente Max", f"{int(ampere_max)} A")
        st.metric("Cavo Consigliato", f"{cavo_rec} mmq")

# --- TAB 5: INVERNO ---
with tab_winter:
    st.subheader("Simulazione Invernale")
    prod_inv = kw_pv_tot * ore_sole_inverno * efficienza_sistema
    diff = prod_inv - fabbisogno_invernale_tot
    cw1, cw2 = st.columns(2)
    cw1.metric("Prod. Inverno", f"{round(prod_inv, 1)} kWh")
    cw2.metric("Bilancio", f"{round(diff, 1)} kWh", delta="OK" if diff>=0 else "LOW")
    
    hours = np.arange(48); soc = []; curr = kwh_batt_tot*0.9; avg = fabbisogno_invernale_tot/24
    for h in hours:
        hd = h%24
        l = avg * (1.5 if 18<=hd<=22 else 0.8)
        p = (prod_inv/6)*(1-abs(12-hd)/3) if 9<=hd<=15 else 0
        curr += (max(0,p)-l)
        if curr>kwh_batt_tot: curr=kwh_batt_tot
        if curr<0: curr=0
        soc.append(curr)
        
    fig_w, ax = plt.subplots(figsize=(10, 3))
    ax.plot(hours, soc, color='green', label='Energia Batteria (kWh)')
    ax.axhline(0, color='red', linestyle='--', label='Blackout (0%)')
    ax.set_ylabel("kWh")
    ax.set_title("Andamento Batteria (48 Ore)")
    ax.legend(loc='upper right', frameon=True)
    ax.grid(True, alpha=0.3)
    st.pyplot(fig_w)

# --- TAB 6: ESTATE ---
with tab_summer:
    st.subheader("Simulazione Estiva")
    prod_est = kw_pv_tot * ore_sole_estate * efficienza_sistema
    kwh_clima=0
    if st.checkbox("Simula Clima", value=True):
        kwh_clima = (600*8)/1000
    surplus_est = prod_est - (fabbisogno_base_kwh + kwh_ev + kwh_clima)
    st.metric("Surplus (Smart Load)", f"{round(surplus_est, 1)} kWh", delta="Gratis")
    
    hs = np.arange(24); ss=[]; ls=[]; bs=[]; cb=kwh_batt_tot*0.3
    base_l = fabbisogno_base_kwh/24
    for h in hs:
        s = np.sin((h-6)*np.pi/14)*(prod_est/8) if 6<=h<=20 else 0
        l = base_l + (0.6 if 13<=h<=17 else 0)
        cb += (max(0,s)-l)
        if cb>kwh_batt_tot: cb=kwh_batt_tot
        if cb<0: cb=0
        ss.append(max(0,s)); ls.append(l); bs.append(cb)
        
    fig_s, ax_s = plt.subplots(figsize=(10, 3))
    # Uniamo le legende
    ln1 = ax_s.plot(hs, ss, color='gold', label='PV')[0]
    ln2 = ax_s.plot(hs, ls, color='blue', label='Carichi')[0]
    ax2 = ax_s.twinx()
    ln3 = ax2.plot(hs, bs, color='green', label='Batt')[0]
    lns = [ln1, ln2, ln3]; labs = [l.get_label() for l in lns]
    ax_s.legend(lns, labs, loc='upper left', frameon=True)
    
    ax_s.set_title("Profilo Giornaliero (24 Ore)")
    ax_s.grid(True, alpha=0.3)
    st.pyplot(fig_s)

# --- TAB 7: RIEPILOGO ---
with tab_report:
    st.subheader("📋 Dashboard Riepilogativa")
    
    st.markdown(f"""
    <div class="summary-box">
    <b>NOME:</b> {client_name} | <b>PROGETTO:</b> {project_name}<br>
    <hr>
    <b>IMPIANTO:</b> {round(kw_pv_tot, 2)} kWp ({totale_pannelli_progetto} Moduli)<br>
    <b>ACCUMULO:</b> {round(kwh_batt_tot, 1)} kWh ({n_b} Batterie)<br>
    <b>INVERTER:</b> {round(picco_potenza_kW, 1)} kW (Consigliato)<br>
    <hr>
    <b>CONFIGURAZIONE TECNICA:</b><br>
    - Pannelli: {pan_watt}W ({pan_w_m}x{pan_h_m}m)<br>
    - Stringa Voc (-10°C): {int(voc_stringa_freddo)} V<br>
    - Cavi Batteria (48V): {cavo_rec} mmq<br>
    - Layout Falde:<br> { '<br>'.join(active_falde_report) }
    <hr>
    <b>PERFORMANCE:</b><br>
    - Autonomia: {round((kwh_batt_tot*0.9)/fabbisogno_invernale_tot, 1) if fabbisogno_invernale_tot>0 else 0} gg<br>
    - Inverno: {round(prod_inv, 1)} kWh (Bilancio: {round(diff, 1)} kWh)<br>
    - Estate: {round(prod_est, 1)} kWh (Surplus: {round(surplus_est, 1)} kWh)<br>
    <hr>
    <div style="font-size: 24px; text-align: right;">
    <b>TOTALE: € {totale_euro:,.2f}</b>
    </div>
    </div>
    """, unsafe_allow_html=True)
    
    txt_down = f"""
    Report OFF-GRID
    Nome: {client_name} - {project_name}
    --------------------------------------
    PV: {kw_pv_tot} kWp ({totale_pannelli_progetto}x {pan_watt}W)
    Batteria: {kwh_batt_tot} kWh
    Inverter: {picco_potenza_kW} kW
    
    COSTO TOTALE: {totale_euro} Euro
    """
    st.download_button("📥 Scarica Report TXT", txt_down, "preventivo.txt")

# --- TAB 8: INFO TECNICHE ---
with tab_info:
    st.subheader("ℹ️ Metodologia di Calcolo")
    st.markdown("""
    Questo software utilizza algoritmi deterministici basati sullo standard industriale PSH (Peak Sun Hours) per il dimensionamento prudenziale di impianti a isola.
    
    ### 1. Produzione Energetica
    L'energia giornaliera media è calcolata come:
    """)
    st.latex(r'''E_{day} = P_{PV} \times PSH \times \eta_{sys}''')
    st.markdown(f"""
    Dove:
    * $P_{{PV}}$: Potenza nominale del campo fotovoltaico ({round(kw_pv_tot, 1)} kWp).
    * $PSH$: Ore di sole picco equivalenti (Input Utente: {ore_sole_inverno}h Inv / {ore_sole_estate}h Est). 
    * $\eta_{{sys}}$: Efficienza complessiva del sistema (Input Utente: {efficienza_sistema*100}%).
    
    ### 2. Simulazione Oraria (Grafici)
    Per generare i profili di carica/scarica, il totale giornaliero viene distribuito sulle 24 ore secondo curve geometriche:
    * **Inverno:** Distribuzione "a campana stretta" (Parabola) concentrata tra le 09:00 e le 15:00.
    * **Estate:** Distribuzione "sinusoidale ampia" dalle 06:00 alle 20:00.
    
    ### 3. Bilancio Batteria
    Lo stato di carica (SoC) viene calcolato iterativamente ogni ora:
    """)
    st.latex(r'''SoC_{t} = SoC_{t-1} + (E_{PV,t} - E_{Load,t})''')
    st.markdown("""
    * Se $SoC > CapacitàMax$: L'energia eccedente è considerata "Surplus" (Taglio produzione).
    * Se $SoC < 0$: Si verifica un Blackout (linea rossa nel grafico).
    """)
    
    st.info("Nota: I calcoli sono stime basate sui dati inseriti. Le condizioni meteo reali possono variare significativamente.")