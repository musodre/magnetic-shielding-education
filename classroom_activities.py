# SPDX-License-Identifier: MIT
"""Runnable classroom studies matching Supplementary Material II.

Example: python classroom_activities.py --mu 1000 --k 0.5 --h0 1 --geometry sphere
All lengths are normalized by b=1; H0 is in A/m. The single configuration
and interface table use the command-line parameters. Benchmark sweeps are
fixed and labelled separately. No notebook or live interface is implied.
"""
from pathlib import Path
import argparse, csv, json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.colors import Normalize
from generate_revision_figures import sf, fields, coefficients, KC, validate

OUT=Path(__file__).resolve().parent
def write_csv(name,header,rows):
    with (OUT/name).open('w',newline='') as f:
        writer=csv.writer(f);writer.writerow(header);writer.writerows(rows)

def approximation_error(mu,k,geometry):
    aa=2/9*(1-k**3) if geometry=='sphere' else (1-k**2)/4
    return (sf(mu,k,geometry)-aa*np.asarray(mu))/sf(mu,k,geometry)

def tolerance_threshold(k,geometry,eta=.05):
    aa=2/9*(1-k**3) if geometry=='sphere' else (1-k**2)/4
    cc=(5+4*k**3)/9 if geometry=='sphere' else (1+k**2)/2
    return ((1-eta)*cc+np.sqrt((1-eta)**2*cc**2+4*eta*(1-eta)*aa**2))/(2*eta*aa)

def interface_rows(mu,k,h0,geometry):
    hi,aa,dd,cc,p=coefficients(mu,k,1,h0,geometry)
    cost=sint=1/np.sqrt(2)
    rows=[]
    for radius,label in [(k,'a'),(1,'b')]:
        hn=(aa-p*dd/radius**(p+1))*cost
        ht=-(aa+dd/radius**(p+1))*sint
        if label=='a':
            cases=[('cavity',hi*cost,-hi*sint,1),('shell',hn,ht,mu)]
        else:
            cases=[('shell',hn,ht,mu),('exterior',(h0+p*cc)*cost,-(h0-cc)*sint,1)]
        for side,n,t,m in cases:rows.append([label,side,n/h0,t/h0,m*n/h0,m*t/h0])
    return rows

def draw_field(ax,mu,k,h0,geometry):
    line=np.linspace(-2,2,401);x,y=np.meshgrid(line,line)
    hx,hy,bn,cav=fields(mu,k,1,h0,geometry,x,y)
    im=ax.pcolormesh(x,y,np.log10(np.maximum(bn,1e-30)),cmap='viridis',norm=Normalize(-2.5,.8),shading='auto',rasterized=True)
    ax.streamplot(line,line,np.ma.array(hx,mask=cav),np.ma.array(hy,mask=cav),density=.5,color='white',linewidth=.5,arrowsize=.6)
    for yy in [-.52*k,0,.52*k]:
        half=np.sqrt(k*k-yy*yy)*.85
        ax.plot([-half,half],[yy,yy],color='white',lw=.8)
        ax.annotate('',xy=(.2*k,yy),xytext=(-.2*k,yy),arrowprops={'arrowstyle':'->','color':'white','lw':.8})
    for radius in [k,1]:ax.add_patch(Circle((0,0),radius,fill=False,ec='k',lw=.8))
    ax.set(xlabel='x/b',ylabel='y/b',aspect='equal',xlim=(-2,2),ylim=(-2,2))
    ax.set_title(rf'$\mu_r={mu:g}$, SF = {sf(mu,k,geometry):.3f}')
    return im

def studies(args):
    validation=validate()
    mu,k,h0,geom=args.mu,args.k,args.h0,args.geometry
    values=interface_rows(mu,k,h0,geom)
    write_csv('interface_values.csv',['interface','side','H_n/H0','H_t/H0','B_n/B0','B_t/B0'],values)
    benchmarks=[[m,sf(m,.5,'sphere'),sf(m,.5,'cylinder'),1/sf(m,.5,'sphere'),1/sf(m,.5,'cylinder')] for m in [1,10,100,1000]]
    write_csv('permeability_sweep.csv',['mu_r','SF_sphere','SF_cylinder','attenuation_sphere','attenuation_cylinder'],benchmarks)
    geometry_rows=[[kk,sf(1000,kk,'sphere'),sf(1000,kk,'cylinder'),sf(1000,kk,'sphere')/sf(1000,kk,'cylinder')] for kk in [.3,KC,.5,.8,.95]]
    write_csv('geometry_sweep.csv',['k','SF_sphere','SF_cylinder','ratio'],geometry_rows)
    thresholds=[[g,kk,tolerance_threshold(kk,g)] for kk in [.5,.95] for g in ['sphere','cylinder']]
    write_csv('approximation_thresholds.csv',['geometry','k','mu_r_at_5_percent_error'],thresholds)
    mm=np.geomspace(1,1e5,501)
    write_csv('approximation_errors.csv',['mu_r','sphere_k0.5','cylinder_k0.5','sphere_k0.95','cylinder_k0.95'],[[m,*[approximation_error(m,kk,g) for kk in [.5,.95] for g in ['sphere','cylinder']]] for m in mm])
    # One-sided interface checks use individual components, not streamline interpolation.
    for offset in [0,2]:
        assert abs(values[offset][4]-values[offset+1][4])<1e-10
        assert abs(values[offset][3]-values[offset+1][3])<1e-10
    assert np.isclose(sf(mu,k,geom),h0/coefficients(mu,k,1,h0,geom)[0],rtol=1e-10)
    for g,kk,threshold in thresholds:
        assert np.isclose(approximation_error(threshold,kk,g),.05,rtol=1e-10)
    report={'configuration':{'mu_r':mu,'k':k,'H0_A_per_m':h0,'geometry':geom},
            'SF':float(sf(mu,k,geom)),'H_cavity_A_per_m':float(h0/sf(mu,k,geom)),
            'H_cavity_over_H0':float(1/sf(mu,k,geom)),
            'crossing_k':float(KC),'validation':validation,'thresholds_5_percent':thresholds}
    (OUT/'classroom_results.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
    fig,ax=plt.subplots(figsize=(5.5,4.6),layout='constrained')
    im=draw_field(ax,mu,k,h0,geom)
    fig.colorbar(im,ax=ax,label=r'$\log_{10}(|\mathbf{B}|/B_0)$',shrink=.9)
    fig.savefig(OUT/'single_configuration.pdf',dpi=300);plt.close(fig)
    fig,axs=plt.subplots(2,2,figsize=(7.2,6.4),layout='constrained')
    for ax,m in zip(axs.ravel(),[1,10,100,1000]):im=draw_field(ax,m,.5,1,'sphere')
    fig.colorbar(im,ax=axs,label=r'$\log_{10}(|\mathbf{B}|/B_0)$',shrink=.8)
    fig.savefig(OUT/'activity_permeability.pdf',dpi=300);plt.close(fig)
    fig,ax=plt.subplots(figsize=(6.4,3.7),layout='constrained')
    for kk,col in [(.5,'#2166ac'),(.95,'#b2182b')]:
        for g,style in [('sphere','-'),('cylinder','--')]:
            ax.loglog(mm,approximation_error(mm,kk,g),style,color=col,lw=1.6,label=f'{g}, k={kk}')
    ax.axhline(.05,color='.3',ls=':',lw=1,label='5% tolerance')
    ax.set(xlabel=r'$\mu_r$',ylabel='Relative asymptotic error',ylim=(1e-5,1.1),xlim=(1,1e5))
    ax.legend(loc='lower left',fontsize=10);ax.grid(alpha=.2)
    fig.savefig(OUT/'activity_approximation.pdf');plt.close(fig)
    # Reference tables always use stated fixed parameters, independently of CLI arguments.
    tex=[r'\begin{center}',r'\begin{tabular}{r r r r}',r'$\mu_r$ & $SF_s$ & $SF_c$ & $h_s/H_0$ \\',r'\hline']
    tex += [f'{m:g} & {s:.4f} & {c:.4f} & {atts:.6f} '+r'\\' for m,s,c,atts,attc in benchmarks]
    tex += [r'\end{tabular}',r'\end{center}']
    (OUT/'table_permeability.tex').write_text('\n'.join(tex)+'\n')
    tex=[r'\begin{center}',r'\begin{tabular}{l r r}',r'Geometry & $k$ & $\mu_r$ at 5\% error \\',r'\hline']
    tex += [f'{g.capitalize()} & {kk:g} & {threshold:.3f} '+r'\\' for g,kk,threshold in thresholds]
    tex += [r'\end{tabular}',r'\end{center}']
    (OUT/'table_thresholds.tex').write_text('\n'.join(tex)+'\n')
    fixed=interface_rows(1000,.5,1,'sphere')
    tex=[r'\begin{center}',r'\begin{tabular}{l l r r r r}',r'Boundary & Side & $H_n/H_0$ & $H_t/H_0$ & $B_n/B_0$ & $B_t/B_0$ \\',r'\hline']
    def scientific(v):
        mantissa,exponent=f'{v:.3e}'.split('e')
        return '$'+mantissa+r'\times10^{'+str(int(exponent))+'}$'
    tex += [f'{bd} & {side} & '+ ' & '.join(scientific(v) for v in nums)+r' \\' for bd,side,*nums in fixed]
    tex += [r'\end{tabular}',r'\end{center}']
    (OUT/'table_interfaces.tex').write_text('\n'.join(tex)+'\n')

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mu',type=float,default=1000.)
    parser.add_argument('--k',type=float,default=.5)
    parser.add_argument('--h0',type=float,default=1.)
    parser.add_argument('--geometry',choices=['sphere','cylinder'],default='sphere')
    args=parser.parse_args()
    if not np.isfinite([args.mu,args.k,args.h0]).all() or args.mu<1 or not 0<args.k<1 or args.h0<=0:
        parser.error('Require finite mu >= 1, 0 < k < 1, H0 > 0.')
    studies(args)
