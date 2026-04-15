"""
Modelo de Potts q=5 con Campo de Pánico — Simulaciones MC
TFG: Luis F. García Lobatón — UCM 2025-26
"""
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from scipy.stats import kurtosis, skew, kstest
from scipy.optimize import curve_fit
import warnings, time, pickle, os
warnings.filterwarnings("ignore")
np.random.seed(42)

plt.rcParams.update({
    "font.size":11,"axes.labelsize":13,"axes.titlesize":11,
    "xtick.labelsize":10,"ytick.labelsize":10,"legend.fontsize":9.5,
    "figure.dpi":150,"font.family":"serif","mathtext.fontset":"cm",
})

def Tc_ex(q): return 1.0/np.log(1.0+np.sqrt(float(q)))

def sweep(sigma, L, beta, Jeff, q):
    N = L*L
    for _ in range(N):
        i, j = np.random.randint(0,L), np.random.randint(0,L)
        s_old = sigma[i,j]
        s_new = np.random.randint(1,q)
        if s_new >= s_old: s_new += 1
        nb = [sigma[(i-1)%L,j],sigma[(i+1)%L,j],sigma[i,(j-1)%L],sigma[i,(j+1)%L]]
        delta = sum(int(s_new==s)-int(s_old==s) for s in nb)
        dH = -Jeff*delta
        if dH<=0 or np.random.rand()<np.exp(-beta*dH): sigma[i,j]=s_new
    return sigma

def mag(sigma,q):
    c=np.bincount(sigma.ravel(),minlength=q+1)[1:q+1]
    return (q*c.max()/sigma.size-1.0)/(q-1)

def phi_max(sigma,q):
    c=np.bincount(sigma.ravel(),minlength=q+1)[1:q+1]
    return c.max()/sigma.size

def indice(sigma,q): return phi_max(sigma,q)-1.0/q

Q=5; Tc=Tc_ex(Q)
print(f"T_c(q={Q}) = {Tc:.6f}")

import sys
which = sys.argv[1] if len(sys.argv)>1 else "all"

# ═══════════ FIG 1 ════════════════════════════════════════════════════
if which in ("1","all"):
    print("\n[Fig 1] ...")
    t0=time.time(); L=12
    rT=np.linspace(0.40*Tc,1.15*Tc,14)
    etas=[0.00,0.20,0.40,0.60]; cols=["#1f77b4","#ff7f0e","#2ca02c","#d62728"]
    fig,ax=plt.subplots(figsize=(7.5,5.5))
    for eta,col in zip(etas,cols):
        Jeff=1.0*(1-eta); ms=[]
        for T in rT:
            s=np.ones((L,L),dtype=int)
            for _ in range(200): s=sweep(s,L,1.0/T,Jeff,Q)
            med=[phi_max(sweep(s,L,1.0/T,Jeff,Q),Q) for _ in range(200)]
            ms.append(np.mean(med))
        ax.axvline(1-eta,color=col,ls=":",lw=1,alpha=0.5)
        ax.plot(rT/Tc,ms,"o-",color=col,ms=5,lw=1.8,
                label=rf"$\eta={eta:.2f}$  ($T_c^{{\rm ef}}/T_c={1-eta:.2f}$)")
    ax.axhline(1.0/Q,color="gray",ls=":",lw=1.2,alpha=0.6)
    ax.text(1.16,1.0/Q+0.01,r"$1/q$",color="gray",fontsize=10,va="bottom")
    ax.set_xlabel(r"$T/T_c$"); ax.set_ylabel(r"Popularidad dominante $\phi_{k^*}$")
    ax.set_title(rf"Diagrama de fases ($q={Q}$, $L={L}$). Punteadas: $T_c^{{\rm ef}}/T_c=1-\eta$")
    ax.legend(loc="upper right"); ax.set_xlim(0.35,1.20); ax.set_ylim(0.15,1.05)
    ax.grid(alpha=0.25); fig.tight_layout()
    fig.savefig("fig1_diagrama_fases.pdf",bbox_inches="tight")
    fig.savefig("fig1_diagrama_fases.png",bbox_inches="tight",dpi=150)
    plt.close(); print(f"  OK ({time.time()-t0:.0f}s)")

# ═══════════ FIG 2 ════════════════════════════════════════════════════
if which in ("2","all"):
    print("\n[Fig 2] ...")
    t0=time.time(); L=12
    rats=[0.70,0.85,0.95]; cols=["#2ca02c","#1f77b4","#d62728"]
    etas2=np.linspace(0,0.99,12)
    fig,ax=plt.subplots(figsize=(7.5,5.5))
    for rT,col in zip(rats,cols):
        beta=1.0/(rT*Tc); ms=[]
        for eta in etas2:
            s=np.ones((L,L),dtype=int); Jeff=1.0*(1-eta)
            for _ in range(200): s=sweep(s,L,beta,Jeff,Q)
            med=[phi_max(sweep(s,L,beta,Jeff,Q),Q) for _ in range(200)]
            ms.append(np.mean(med))
        ax.axvline(1-rT,color=col,ls=":",lw=1,alpha=0.5)
        ax.plot(etas2,ms,"o-",color=col,ms=5,lw=1.8,
                label=rf"$T/T_c={rT:.2f}$  ($\eta_c={1-rT:.2f}$)")
    ax.axhline(1.0/Q,color="gray",ls=":",lw=1.2,alpha=0.6)
    ax.text(0.97,1.0/Q+0.01,r"$1/q$",color="gray",fontsize=10,va="bottom",ha="right")
    ax.set_xlabel(r"$\eta$"); ax.set_ylabel(r"Popularidad dominante $\phi_{k^*}$")
    ax.set_title(rf"Colapso inducido por pánico ($q={Q}$, $L={L}$). Punteadas: $\eta_c=1-T/T_c$")
    ax.legend(); ax.set_xlim(-0.02,1.02); ax.set_ylim(0.15,1.05); ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig("fig2_colapso_eta.pdf",bbox_inches="tight")
    fig.savefig("fig2_colapso_eta.png",bbox_inches="tight",dpi=150)
    plt.close(); print(f"  OK ({time.time()-t0:.0f}s)")

# ═══════════ FIG 3 ════════════════════════════════════════════════════
if which in ("3","all"):
    print("\n[Fig 3] ...")
    t0=time.time(); L=40
    cmap5=ListedColormap(["#1f77b4","#ff7f0e","#2ca02c","#d62728","#9467bd"])
    fig,axes=plt.subplots(2,4,figsize=(14,7))
    for k,rT in enumerate([0.50,0.75,0.90,1.10]):
        s=np.ones((L,L),dtype=int)
        for _ in range(350): s=sweep(s,L,1.0/(rT*Tc),1.0,Q)
        axes[0,k].imshow(s,cmap=cmap5,vmin=1,vmax=Q,interpolation="nearest")
        axes[0,k].set_title(rf"$T/T_c={rT:.2f}$, $\eta=0$",fontsize=10)
        axes[0,k].axis("off")
    # Fila inferior: shock
    s=np.ones((L,L),dtype=int)
    for _ in range(350): s=sweep(s,L,1.0/(0.85*Tc),1.0,Q)
    axes[1,0].imshow(s.copy(),cmap=cmap5,vmin=1,vmax=Q,interpolation="nearest")
    axes[1,0].set_title("Pre-shock\n"+r"$T/T_c=0.85$",fontsize=10); axes[1,0].axis("off")
    Js=1.0*0.40; bs=1.0/(0.85*Tc)
    for k,nt,lab in [(1,15,r"$t=15$"),(2,45,r"$t=60$"),(3,190,r"$t=250$")]:
        for _ in range(nt): s=sweep(s,L,bs,Js,Q)
        axes[1,k].imshow(s,cmap=cmap5,vmin=1,vmax=Q,interpolation="nearest")
        axes[1,k].set_title(lab+" PMC post-shock\n"+r"$\eta=0.60$",fontsize=10)
        axes[1,k].axis("off")
    fig.suptitle(rf"Instantáneas ($q={Q}$, $L={L}$). Arriba: equilibrio. Abajo: colapso $\eta=0.60$.",
                 fontsize=11,y=1.02)
    fig.tight_layout()
    fig.savefig("fig3_snapshots.pdf",bbox_inches="tight")
    fig.savefig("fig3_snapshots.png",bbox_inches="tight",dpi=150)
    plt.close(); print(f"  OK ({time.time()-t0:.0f}s)")

# ═══════════ FIG 4 ════════════════════════════════════════════════════
if which in ("4","all"):
    print("\n[Fig 4] ...")
    t0=time.time(); L=14; T4=0.88*Tc; b4=1.0/T4; nt=1500; tsh=750
    fig,axes=plt.subplots(2,1,figsize=(11,7),sharex=True)
    for idx,(es,tit,col) in enumerate([
        (0.0,r"Sin pánico ($\eta=0$)","#1f77b4"),
        (0.65,r"Shock $\eta=0.65$ en $t_{\rm shock}$","#d62728")]):
        s=np.ones((L,L),dtype=int)
        for _ in range(250): s=sweep(s,L,b4,1.0,Q)
        ser=[]
        for t in range(nt):
            Jeff=1.0 if(t<tsh or es==0)else 1.0*(1-es)
            s=sweep(s,L,b4,Jeff,Q); ser.append(indice(s,Q))
        ser=np.array(ser); ax=axes[idx]
        ax.plot(range(nt),ser,lw=0.6,color=col,alpha=0.9)
        if es>0:
            ax.axvline(tsh,color="k",ls="--",lw=1.5,label=rf"$t_{{\rm shock}}={tsh}$")
            ax.legend(fontsize=10)
        ax.set_ylabel(r"$\mathcal{I}(t)$"); ax.set_title(tit,fontsize=10)
        ax.set_ylim(-0.02,0.85); ax.grid(alpha=0.2)
        print(f"  η={es}: <I>_antes={ser[:tsh].mean():.4f} <I>_después={ser[tsh:].mean():.4f}")
    axes[1].set_xlabel("Tiempo (PMC)")
    fig.suptitle(rf"Dinámica de crisis ($q={Q}$, $L={L}$, $T=0.88\,T_c$)",fontsize=11,y=1.01)
    fig.tight_layout()
    fig.savefig("fig4_shock_lehman.pdf",bbox_inches="tight")
    fig.savefig("fig4_shock_lehman.png",bbox_inches="tight",dpi=150)
    plt.close(); print(f"  OK ({time.time()-t0:.0f}s)")

# ═══════════ FIG 5 ════════════════════════════════════════════════════
if which in ("5","all"):
    print("\n[Fig 5] ...")
    t0=time.time(); L=12; T5=0.90*Tc; b5=1.0/T5
    # Distribuciones para umbral
    nU=400
    phi_b=[]
    s=np.ones((L,L),dtype=int)
    for _ in range(150): s=sweep(s,L,b5,1.0,Q)
    for _ in range(nU): s=sweep(s,L,b5,1.0,Q); phi_b.append(phi_max(s,Q))
    phi_d=[]
    s=np.random.randint(1,Q+1,size=(L,L))
    Jd=1.0*0.30
    for _ in range(150): s=sweep(s,L,b5,Jd,Q)
    for _ in range(nU): s=sweep(s,L,b5,Jd,Q); phi_d.append(phi_max(s,Q))
    phi_b,phi_d=np.array(phi_b),np.array(phi_d)
    mu_d,sig_d=phi_d.mean(),phi_d.std()
    phi_c=mu_d+2*sig_d
    print(f"  φ_bub={phi_b.mean():.3f}±{phi_b.std():.3f}")
    print(f"  φ_dis={mu_d:.3f}±{sig_d:.3f}")
    print(f"  φ_c={phi_c:.3f}")

    etas5=[0.00,0.10,0.20,0.35,0.50,0.65,0.80]; nr=15; tmax=1500
    durs={}
    for eta in etas5:
        Jeff=1.0*(1-eta); dl=[]
        for _ in range(nr):
            s=np.ones((L,L),dtype=int); ok=False
            for t in range(1,tmax+1):
                s=sweep(s,L,b5,Jeff,Q)
                if t%5==0 and phi_max(s,Q)<phi_c:
                    dl.append(t); ok=True; break
            if not ok: dl.append(tmax)
        durs[eta]=np.array(dl)
        print(f"  η={eta:.2f}: <τ>={np.mean(dl):.0f}±{np.std(dl)/np.sqrt(nr):.0f}")

    fig,(a,b,c)=plt.subplots(1,3,figsize=(15,5))
    a.hist(phi_b,bins=22,density=True,alpha=0.6,color="#1f77b4",
           label=rf"Burbuja: $\bar\phi={phi_b.mean():.3f}$")
    a.hist(phi_d,bins=22,density=True,alpha=0.6,color="#d62728",
           label=rf"Desordenada: $\bar\phi={mu_d:.3f}$")
    a.axvline(phi_c,color="k",ls="--",lw=2,label=rf"$\phi_c={phi_c:.3f}$")
    a.axvline(1.0/Q,color="gray",ls=":",lw=1.5,label=r"$1/q$")
    a.set_xlabel(r"$\phi_{k^*}$"); a.set_ylabel("Densidad")
    a.set_title(r"(a) Distribuciones de $\phi_{k^*}$"); a.legend(fontsize=8); a.grid(alpha=0.2)

    bp=b.boxplot([durs[e] for e in etas5],labels=[f"{e:.2f}" for e in etas5],
                 patch_artist=True,medianprops=dict(color="black",lw=2))
    cm=plt.cm.RdYlGn_r(np.linspace(0.1,0.9,len(etas5)))
    for bx,co in zip(bp["boxes"],cm): bx.set_facecolor(co); bx.set_alpha(0.7)
    b.set_xlabel(r"$\eta$"); b.set_ylabel("Duración (PMC)")
    b.set_title(f"(b) Distribución ({nr} realizaciones)"); b.grid(alpha=0.25,axis="y")

    med5=np.array([np.mean(durs[e]) for e in etas5])
    sem5=np.array([np.std(durs[e])/np.sqrt(nr) for e in etas5])
    ea=np.array(etas5)
    c.errorbar(ea,med5,yerr=sem5,fmt="o",color="#d62728",capsize=5,ms=7,lw=2,
               label=r"$\langle\tau\rangle\pm$SEM")
    try:
        def exd(eta,t0,g): return t0*np.exp(-g*eta)
        mf=(med5<0.95*tmax)&(med5>6)
        po,pc=curve_fit(exd,ea[mf],med5[mf],sigma=np.clip(sem5[mf],1,None),absolute_sigma=True,p0=[800,3],maxfev=5000)
        pe=np.sqrt(np.diag(pc))
        pe=np.where(np.isfinite(pe),pe,0)
        ep=np.linspace(0,0.85,200)
        c.plot(ep,exd(ep,*po),"--",color="#1f77b4",lw=2,
               label=rf"$\tau_0\,e^{{-\gamma\eta}}$: $\gamma={po[1]:.2f}$")
        print(f"  Ajuste: tau0={po[0]:.0f}, gamma={po[1]:.2f}")
    except Exception as e: print(f"  Ajuste error: {e}")
    c.set_yscale("log"); c.set_xlabel(r"$\eta$"); c.set_ylabel(r"$\langle\tau\rangle$ (log)")
    c.set_title(rf"(c) $\langle\tau\rangle$ vs $\eta$"); c.legend(fontsize=8)
    c.grid(alpha=0.25,which="both")
    fig.tight_layout()
    fig.savefig("fig5_lifetimes.pdf",bbox_inches="tight")
    fig.savefig("fig5_lifetimes.png",bbox_inches="tight",dpi=150)
    plt.close()
    # Save data for LaTeX
    pickle.dump({"phi_b":phi_b,"phi_d":phi_d,"phi_c":phi_c,
                 "durs":durs,"etas":etas5},open("fig5_data.pkl","wb"))
    print(f"  OK ({time.time()-t0:.0f}s)")

# ═══════════ FIG 6 ════════════════════════════════════════════════════
if which in ("6","all"):
    print("\n[Fig 6] ...")
    t0=time.time(); L=16; T6=0.96*Tc; b6=1.0/T6; n6=3500
    escs=[(0.0,"$\\eta=0$","#1f77b4"),(0.25,"$\\eta=0.25$","#ff7f0e"),
          (0.50,"$\\eta=0.50$","#d62728")]
    ret_e={}
    for eta,lab,col in escs:
        Jeff=1.0*(1-eta)
        s=np.random.randint(1,Q+1,size=(L,L))
        for _ in range(350): s=sweep(s,L,b6,Jeff,Q)
        ser=[indice(sweep(s,L,b6,Jeff,Q),Q) for _ in range(n6)]
        ret=np.diff(np.array(ser)); ret_e[eta]=ret
        rs=(ret-ret.mean())/max(ret.std(),1e-12)
        _,pk=kstest(rs,"norm")
        print(f"  η={eta:.2f}: κ={kurtosis(ret):.3f} p-KS={pk:.2e}")

    fig,(a,b)=plt.subplots(1,2,figsize=(12,5.5))
    for eta,lab,col in escs:
        r=ret_e[eta]; sr=r.std()
        a.hist(r,bins=np.linspace(-5*sr,5*sr,65),density=True,alpha=0.4,color=col,
               label=lab+rf" ($\kappa={kurtosis(r):.2f}$)")
    r0=ret_e[0.0]; s0=r0.std()
    xg=np.linspace(-5*s0,5*s0,300)
    a.plot(xg,np.exp(-0.5*(xg/s0)**2)/(s0*np.sqrt(2*np.pi)),"k--",lw=2,label="Gaussiana")
    a.set_xlabel(r"$r_{\mathcal{I}}$"); a.set_ylabel("Densidad")
    a.set_title("Distribución de retornos"); a.legend(fontsize=8); a.grid(alpha=0.2)

    mu_exp=3.0
    for eta,lab,col in escs:
        r=ret_e[eta]; ar=np.sort(np.abs(r))
        cc=1.0-np.arange(1,len(ar)+1)/float(len(ar))
        m=(ar>1e-5)&(cc>0.005)
        b.loglog(ar[m],cc[m],"-",color=col,lw=1.8,alpha=0.85,label=lab)
    ar0=np.sort(np.abs(ret_e[0.0]))
    cc0=1.0-np.arange(1,len(ar0)+1)/float(len(ar0))
    p80=ar0[int(0.80*len(ar0))]
    mt=(ar0>p80)&(cc0>0.005)&(ar0>1e-5)
    if mt.sum()>5:
        try:
            def plw(x,a,m): return a*x**(-m)
            pp,_=curve_fit(plw,ar0[mt],cc0[mt],p0=[0.01,3],maxfev=5000)
            mu_exp=pp[1]; xf=np.linspace(ar0[mt].min(),ar0[mt].max(),100)
            b.loglog(xf,plw(xf,*pp),"k--",lw=2,label=rf"$\mu={mu_exp:.2f}$")
            print(f"  μ={mu_exp:.3f}")
        except: pass
    b.set_xlabel(r"$|r|$"); b.set_ylabel(r"$P(|r|>x)$")
    b.set_title("CCDF log-log"); b.legend(fontsize=8); b.grid(alpha=0.25,which="both")
    fig.suptitle(rf"Colas pesadas ($q={Q}$, $L={L}$, $T=0.96\,T_c$, {n6} PMC)",
                 fontsize=11,y=1.01)
    fig.tight_layout()
    fig.savefig("fig6_retornos_colas.pdf",bbox_inches="tight")
    fig.savefig("fig6_retornos_colas.png",bbox_inches="tight",dpi=150)
    plt.close()
    pickle.dump({"mu":mu_exp,"ret":ret_e},open("fig6_data.pkl","wb"))
    print(f"  OK ({time.time()-t0:.0f}s)")

# ═══════════ FIG 7 ════════════════════════════════════════════════════
if which in ("7","all"):
    print("\n[Fig 7] ...")
    t0=time.time(); L=16; T7=0.96*Tc; b7=1.0/T7; n7=4000
    s=np.random.randint(1,Q+1,size=(L,L))
    for _ in range(350): s=sweep(s,L,b7,1.0,Q)
    ser7=[indice(sweep(s,L,b7,1.0,Q),Q) for _ in range(n7)]
    ret7=np.diff(np.array(ser7)); vol7=np.abs(ret7)
    kur7=kurtosis(ret7); skw7=skew(ret7)
    print(f"  κ={kur7:.3f} skew={skw7:.4f}")

    lmax=70
    mr=ret7.mean(); vr=max(((ret7-mr)**2).mean(),1e-15)
    mv=vol7.mean(); vv=max(((vol7-mv)**2).mean(),1e-15)
    acf_r=np.array([((ret7[:len(ret7)-k]-mr)*(ret7[k:]-mr)).mean()/vr for k in range(lmax)])
    acf_v=np.array([((vol7[:len(vol7)-k]-mv)*(vol7[k:]-mv)).mean()/vv for k in range(lmax)])
    print(f"  ACF(r,1)={acf_r[1]:.4f} ACF(|r|,1)={acf_v[1]:.4f} ACF(|r|,20)={acf_v[20]:.4f}")

    rp=np.sort(ret7[ret7>0]); rn=np.sort(np.abs(ret7[ret7<0]))
    fig,ax=plt.subplots(2,2,figsize=(13,9))

    sg=ret7.std()
    ax[0,0].hist(ret7,bins=np.linspace(-5*sg,5*sg,65),density=True,color="#1f77b4",alpha=0.6,
                 label=rf"MC ($\kappa={kur7:.2f}$)")
    xg=np.linspace(-5*sg,5*sg,300)
    ax[0,0].plot(xg,np.exp(-0.5*(xg/sg)**2)/(sg*np.sqrt(2*np.pi)),"r-",lw=2,label="Gaussiana")
    ax[0,0].set_xlabel("$r$"); ax[0,0].set_ylabel("Densidad")
    ax[0,0].set_title("(a) Colas pesadas"); ax[0,0].legend(); ax[0,0].grid(alpha=0.2)

    ic=1.96/np.sqrt(len(ret7))
    ax[0,1].bar(np.arange(1,50),acf_r[1:50],color="#1f77b4",alpha=0.7,width=0.8)
    ax[0,1].axhline(ic,color="r",ls="--",lw=1.3,label="IC 95%")
    ax[0,1].axhline(-ic,color="r",ls="--",lw=1.3)
    ax[0,1].axhline(0,color="k",lw=0.8)
    ax[0,1].set_xlabel(r"$\tau$"); ax[0,1].set_ylabel(r"ACF($r_t$)")
    ax[0,1].set_title(r"(b) ACF retornos"); ax[0,1].legend()
    ax[0,1].set_ylim(-0.15,0.30); ax[0,1].grid(alpha=0.2)

    lc=np.arange(1,lmax)
    ax[1,0].plot(lc,acf_v[1:lmax],"o-",color="#ff7f0e",ms=3,lw=1,alpha=0.8,label=r"ACF($|r_t|$)")
    ax[1,0].axhline(0,color="k",lw=0.8)
    ax[1,0].axhline(ic,color="r",ls="--",lw=1.3)
    ax[1,0].axhline(-ic,color="r",ls="--",lw=1.3)
    gv=0.30
    mp=acf_v[2:lmax]>0.003; lf=lc[1:][mp]; af=acf_v[2:lmax][mp]
    if len(lf)>5:
        try:
            def pla(t,a,g): return a*t**(-g)
            ppv,_=curve_fit(pla,lf,af,p0=[0.3,0.3],maxfev=5000)
            gv=ppv[1]; tp=np.linspace(max(2,lf.min()),lf.max(),200)
            ax[1,0].plot(tp,pla(tp,*ppv),"--",color="#2ca02c",lw=2,label=rf"$\gamma={gv:.2f}$")
            print(f"  γ={gv:.3f}")
        except: pass
    ax[1,0].set_xlabel(r"$\tau$"); ax[1,0].set_ylabel(r"ACF($|r_t|$)")
    ax[1,0].set_title("(c) Agrupamiento volatilidad"); ax[1,0].legend(); ax[1,0].grid(alpha=0.2)

    np_,nn_=len(rp),len(rn)
    ax[1,1].semilogy(rp,1.0-np.arange(1,np_+1)/float(np_),"-",color="#2ca02c",lw=2,label="$r>0$")
    ax[1,1].semilogy(rn,1.0-np.arange(1,nn_+1)/float(nn_),"-",color="#d62728",lw=2,label="$|r|$ ($r<0$)")
    ax[1,1].set_xlabel("$|r|$"); ax[1,1].set_ylabel("$P(|r|>x)$")
    ax[1,1].set_title("(d) Asimetría gan.-pérd."); ax[1,1].legend()
    ax[1,1].grid(alpha=0.2,which="both")

    tha=np.percentile(np.abs(ret7),90)
    nc=max(np.sum(ret7<-tha),1); ns=max(np.sum(ret7>tha),1)
    ra=nc/ns; print(f"  Ratio asim={ra:.3f}")

    fig.suptitle(rf"Hechos estilizados ($q={Q}$, $L={L}$, $T=0.96\,T_c$, {n7} PMC)",
                 fontsize=11,y=1.01)
    fig.tight_layout()
    fig.savefig("fig7_stylized_facts.pdf",bbox_inches="tight")
    fig.savefig("fig7_stylized_facts.png",bbox_inches="tight",dpi=150)
    plt.close()
    pickle.dump({"kur":kur7,"skew":skw7,"acf_r":acf_r,"acf_v":acf_v,
                 "gv":gv,"ra":ra,"ret":ret7},open("fig7_data.pkl","wb"))
    print(f"  OK ({time.time()-t0:.0f}s)")

print("\nFIN")
