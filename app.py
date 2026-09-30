import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="Navier-Stokes Collapsing Vortex", page_icon="🌀", layout="wide")

st.title("Navier-Stokes Collapsing-Vortex Prototype")
st.caption("A numerical reconstruction of the published bounded-energy concentration mechanism. This is a manufactured-solution demonstrator, not an independent proof of finite-time singularity.")

with st.sidebar:
    st.header("Model controls")
    t = st.slider("Time, t", 0.00, 0.995, 0.90, 0.005)
    nu = st.number_input("Kinematic viscosity, nu", min_value=1e-4, max_value=1.0, value=0.02, format="%.4f")
    h = st.slider("Anisotropy exponent, h", 0.001, 0.009, 0.005, 0.001)
    swirl_gain = st.slider("Swirl multiplier", 0.1, 3.0, 1.0, 0.1)
    axial_gain = st.slider("Meridional-flow multiplier", 0.1, 3.0, 1.0, 0.1)
    asymmetry = st.slider("Axial asymmetry", -0.30, 0.30, 0.08, 0.01)
    pulse_amp = st.slider("Annular pulse amplitude", 0.0, 1.0, 0.35, 0.05)
    pulse_modes = st.slider("Pulse azimuthal mode", 2, 24, 10, 1)
    n = st.select_slider("Grid resolution", options=[65, 81, 101, 121, 151], value=101)
    extent_factor = st.slider("Domain / core scale", 3.0, 10.0, 6.0, 0.5)
    st.divider()
    st.markdown("**Interpretation**")
    st.markdown("As `t -> 1`, the core contracts while the peak velocity grows. Numerical results very near `t = 1` are grid-dependent.")

def gradients(a, dr, dz):
    da_dr = np.gradient(a, dr, axis=1, edge_order=2)
    da_dz = np.gradient(a, dz, axis=0, edge_order=2)
    return da_dr, da_dz

def fields(t, n, h, swirl_gain, axial_gain, asymmetry, extent_factor):
    tau = max(1.0 - t, 1e-6)
    lr = tau**0.5
    lz = tau**(0.5-h)
    rmax = extent_factor * lr
    zmax = extent_factor * lz
    r = np.linspace(max(rmax/(n-1)*0.5, 1e-7), rmax, n)
    z = np.linspace(-zmax, zmax, n)
    dr, dz = r[1]-r[0], z[1]-z[0]
    R, Z = np.meshgrid(r, z)
    xi, eta = R/lr, Z/lz
    q = xi**2 + eta**2
    envelope = np.exp(-q)

    # Axisymmetric Stokes streamfunction. This makes the meridional field divergence-free analytically.
    psi_scale = axial_gain * tau**(0.5-h)
    psi = psi_scale * R**2 * envelope * (1.0 + asymmetry*eta)
    dpsi_dr, dpsi_dz = gradients(psi, dr, dz)
    ur = -dpsi_dz / R
    uz = dpsi_dr / R

    # Smooth swirl, zero on the axis, with target amplitude proportional to tau^(-1/2-h).
    utheta = swirl_gain * tau**(-0.5-h) * xi * np.exp(-0.5*q)

    # A smooth pressure ansatz. The exact manufactured force is computed from the resulting imbalance.
    p = -0.5 * (swirl_gain*tau**(-0.5-h))**2 * np.exp(-q)
    return r, z, R, Z, ur, utheta, uz, p, lr, lz, tau, dr, dz

def cylindrical_audits(t, n, h, swirl_gain, axial_gain, asymmetry, extent_factor, nu):
    r,z,R,Z,ur,ut,uz,p,lr,lz,tau,dr,dz = fields(t,n,h,swirl_gain,axial_gain,asymmetry,extent_factor)
    dur_dr,dur_dz = gradients(ur,dr,dz)
    dut_dr,dut_dz = gradients(ut,dr,dz)
    duz_dr,duz_dz = gradients(uz,dr,dz)
    dp_dr,dp_dz = gradients(p,dr,dz)
    div = dur_dr + ur/R + duz_dz

    dt = min(2e-4, max(1e-7, 0.02*tau))
    tm = max(0.0, t-dt)
    tp = min(0.999999, t+dt)
    fm = fields(tm,n,h,swirl_gain,axial_gain,asymmetry,extent_factor)
    fp = fields(tp,n,h,swirl_gain,axial_gain,asymmetry,extent_factor)
    # Interpolation is avoided because the moving domain uses the same normalized grid; values correspond pointwise in similarity coordinates.
    dur_dt = (fp[4]-fm[4])/(tp-tm)
    dut_dt = (fp[5]-fm[5])/(tp-tm)
    duz_dt = (fp[6]-fm[6])/(tp-tm)

    def scalar_lap(a):
        ar,az = gradients(a,dr,dz)
        arr,_ = gradients(ar,dr,dz)
        _,azz = gradients(az,dr,dz)
        return arr + ar/R + azz
    lap_ur = scalar_lap(ur) - ur/R**2
    lap_ut = scalar_lap(ut) - ut/R**2
    lap_uz = scalar_lap(uz)

    res_r = dur_dt + ur*dur_dr + uz*dur_dz - ut**2/R + dp_dr - nu*lap_ur
    res_t = dut_dt + ur*dut_dr + uz*dut_dz + ur*ut/R - nu*lap_ut
    res_z = duz_dt + ur*duz_dr + uz*duz_dz + dp_dz - nu*lap_uz
    speed = np.sqrt(ur**2+ut**2+uz**2)
    vort_theta = dur_dz-duz_dr
    vort_r = -dut_dz
    vort_z = dut_dr+ut/R
    vort = np.sqrt(vort_r**2+vort_theta**2+vort_z**2)
    force = np.sqrt(res_r**2+res_t**2+res_z**2)
    weights = 2*np.pi*R*dr*dz
    energy = 0.5*np.sum(speed**2*weights)
    l2 = np.sqrt(np.sum(speed**2*weights))
    div_l2 = np.sqrt(np.sum(div**2*weights))
    return locals()

D = cylindrical_audits(t,n,h,swirl_gain,axial_gain,asymmetry,extent_factor,nu)

m1,m2,m3,m4,m5 = st.columns(5)
m1.metric("Time to collapse, tau", f"{D['tau']:.3e}")
m2.metric("Core radius, lr", f"{D['lr']:.3e}")
m3.metric("Peak speed", f"{np.max(D['speed']):.3e}")
m4.metric("Kinetic energy", f"{D['energy']:.3e}")
m5.metric("Continuity L2", f"{D['div_l2']:.3e}")

st.warning("The displayed force is the manufactured forcing required by this smooth trial field. It illustrates the proof strategy but is not the full multi-scale forcing from the analytical construction.")

tab1,tab2,tab3,tab4,tab5 = st.tabs(["Core flow", "3D view", "Residual audit", "Pulse mechanism", "Scaling study"])

with tab1:
    c1,c2 = st.columns(2)
    with c1:
        fig = go.Figure(go.Contour(x=D['r'],y=D['z'],z=D['speed'],colorscale="Turbo",colorbar_title="|u|",contours=dict(showlabels=False)))
        skip=max(1,n//24)
        rr=D['R'][::skip,::skip]; zz=D['Z'][::skip,::skip]
        urq=D['ur'][::skip,::skip]; uzq=D['uz'][::skip,::skip]
        scale=0.18*max(D['r'][-1],2*D['z'][-1])/(np.nanmax(np.sqrt(urq**2+uzq**2))+1e-12)
        for i in range(rr.shape[0]):
            for j in range(rr.shape[1]):
                fig.add_annotation(x=rr[i,j]+scale*urq[i,j],y=zz[i,j]+scale*uzq[i,j],ax=rr[i,j],ay=zz[i,j],xref="x",yref="y",axref="x",ayref="y",showarrow=True,arrowhead=2,arrowsize=1,arrowwidth=1,arrowcolor="rgba(255,255,255,0.65)")
        fig.update_layout(title="Meridional flow over speed magnitude",xaxis_title="r",yaxis_title="z",height=620)
        st.plotly_chart(fig,use_container_width=True)
    with c2:
        fig2=make_subplots(rows=2,cols=1,subplot_titles=("Azimuthal velocity", "Vorticity magnitude"))
        fig2.add_trace(go.Heatmap(x=D['r'],y=D['z'],z=D['ut'],colorscale="RdBu",zmid=0,colorbar=dict(title="uθ",y=0.78,len=0.38)),row=1,col=1)
        fig2.add_trace(go.Heatmap(x=D['r'],y=D['z'],z=D['vort'],colorscale="Magma",colorbar=dict(title="|ω|",y=0.22,len=0.38)),row=2,col=1)
        fig2.update_xaxes(title_text="r",row=2,col=1); fig2.update_yaxes(title_text="z",row=1,col=1); fig2.update_yaxes(title_text="z",row=2,col=1)
        fig2.update_layout(height=620)
        st.plotly_chart(fig2,use_container_width=True)

with tab2:
    stride=max(1,n//35)
    rr=D['R'][::stride,::stride]; zz=D['Z'][::stride,::stride]
    theta=np.linspace(0,2*np.pi,36,endpoint=False)
    # Show three azimuthal sheets to keep rendering responsive.
    fig3=go.Figure()
    for th in [0,2*np.pi/3,4*np.pi/3]:
        x=rr*np.cos(th); y=rr*np.sin(th)
        val=D['speed'][::stride,::stride]
        fig3.add_trace(go.Surface(x=x,y=y,z=zz,surfacecolor=val,colorscale="Turbo",showscale=(th==0),opacity=0.78,colorbar=dict(title="|u|")))
    fig3.update_layout(title="Three-dimensional cylindrical slices",scene=dict(xaxis_title="x",yaxis_title="y",zaxis_title="z",aspectmode="data"),height=700)
    st.plotly_chart(fig3,use_container_width=True)

with tab3:
    c1,c2=st.columns(2)
    with c1:
        fig4=go.Figure(go.Heatmap(x=D['r'],y=D['z'],z=D['div'],colorscale="RdBu",zmid=0,colorbar_title="div u"))
        fig4.update_layout(title="Discrete continuity residual",xaxis_title="r",yaxis_title="z",height=520)
        st.plotly_chart(fig4,use_container_width=True)
    with c2:
        fig5=go.Figure(go.Heatmap(x=D['r'],y=D['z'],z=np.log10(D['force']+1e-16),colorscale="Viridis",colorbar_title="log10 |f|"))
        fig5.update_layout(title="Manufactured-force magnitude",xaxis_title="r",yaxis_title="z",height=520)
        st.plotly_chart(fig5,use_container_width=True)
    st.markdown("The momentum residual before forcing is `f = du/dt + (u.grad)u - nu Laplacian(u) + grad(p)`. Defining the forcing by this residual makes the manufactured field an exact continuum solution. The plotted error primarily reflects finite-difference and moving-grid approximations.")

with tab4:
    th=np.linspace(0,2*np.pi,720)
    sigma=0.55
    r0=2.2*D['lr']
    wr=pulse_amp*np.cos(pulse_modes*th)
    wt=pulse_amp*sigma*np.cos(pulse_modes*th)
    stress=wr*wt
    c1,c2=st.columns(2)
    with c1:
        fig6=go.Figure()
        fig6.add_trace(go.Scatter(x=th,y=wr,name="w_r"))
        fig6.add_trace(go.Scatter(x=th,y=wt,name="w_theta"))
        fig6.add_trace(go.Scatter(x=th,y=stress,name="w_r w_theta",line=dict(width=3)))
        fig6.add_hline(y=np.mean(stress),line_dash="dash",annotation_text=f"mean stress = {np.mean(stress):.3f}")
        fig6.update_layout(title="Zero-mean pulse, nonzero quadratic stress",xaxis_title="theta",height=500)
        st.plotly_chart(fig6,use_container_width=True)
    with c2:
        x=(r0+0.22*D['lr']*wr)*np.cos(th); y=(r0+0.22*D['lr']*wr)*np.sin(th)
        fig7=go.Figure(go.Scatter(x=x,y=y,mode="lines",line=dict(color=stress,colorscale="RdBu",width=5),name="pulse ring"))
        fig7.update_layout(title="Illustrative annular oscillatory ring",xaxis=dict(scaleanchor="y",title="x"),yaxis_title="y",height=500)
        st.plotly_chart(fig7,use_container_width=True)
    st.info("This tab demonstrates the key averaging idea: the perturbation velocity can have zero angular mean while its quadratic momentum flux has a nonzero mean. It is not the paper's complete pulse construction.")

with tab5:
    nt=42
    ts=np.linspace(0.1,0.99,nt)
    peaks=[]; energies=[]; lrs=[]; divs=[]
    nstudy=min(n,81)
    for ti in ts:
        X=cylindrical_audits(ti,nstudy,h,swirl_gain,axial_gain,asymmetry,extent_factor,nu)
        peaks.append(np.max(X['speed'])); energies.append(X['energy']); lrs.append(X['lr']); divs.append(X['div_l2'])
    tauv=1-ts
    fig8=make_subplots(rows=2,cols=2,subplot_titles=("Peak speed", "Kinetic energy", "Core radius", "Continuity residual"))
    for vals,row,col,name in [(peaks,1,1,"max |u|"),(energies,1,2,"E"),(lrs,2,1,"lr"),(divs,2,2,"||div u||2")]:
        fig8.add_trace(go.Scatter(x=tauv,y=vals,mode="lines+markers",name=name),row=row,col=col)
        fig8.update_xaxes(type="log",title_text="tau = 1-t",row=row,col=col)
        fig8.update_yaxes(type="log",row=row,col=col)
    fig8.update_layout(height=720,showlegend=False,title="Grid-dependent scaling study")
    st.plotly_chart(fig8,use_container_width=True)
    # Robust log-log slopes over the last half, avoiding zero values.
    sl=slice(nt//2,None)
    peak_slope=np.polyfit(np.log(tauv[sl]),np.log(np.asarray(peaks)[sl]),1)[0]
    energy_slope=np.polyfit(np.log(tauv[sl]),np.log(np.asarray(energies)[sl]),1)[0]
    c1,c2,c3=st.columns(3)
    c1.metric("Fitted peak-speed exponent",f"{peak_slope:.3f}",help="Expected to be near a negative value for growth as tau decreases.")
    c2.metric("Fitted energy exponent",f"{energy_slope:.3f}",help="Positive means core energy decreases as tau approaches zero.")
    c3.metric("Target swirl exponent",f"{-0.5-h:.3f}")

with st.expander("Model equations and limitations"):
    st.markdown(r"""
The prototype uses an axisymmetric Stokes streamfunction, so the meridional field satisfies

$$u_r=-\frac{1}{r}\frac{\partial\psi}{\partial z},\qquad
u_z=\frac{1}{r}\frac{\partial\psi}{\partial r}.$$

The similarity scales are

$$\ell_r=(1-t)^{1/2},\qquad \ell_z=(1-t)^{1/2-h},$$

and the swirl amplitude is proportional to $(1-t)^{-1/2-h}$. The program calculates cylindrical continuity, vorticity, kinetic energy, and the manufactured momentum forcing.

**Limitations:** the trial profiles are pedagogical approximations; the exact proof uses high-order corrections and multi-scale oscillations. The present moving similarity grid is not a conservative adaptive CFD mesh, and finite differences become unreliable if the physical core is no longer resolved.
""")
