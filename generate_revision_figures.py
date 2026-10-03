# SPDX-License-Identifier: MIT
"""Generate vector figures and verify the classical analytical implementation.
Run from this directory: python generate_revision_figures.py
Requires numpy and matplotlib. No notebook is supplied.
"""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Arc
from matplotlib.colors import Normalize

OUT = Path(__file__).resolve().parent
plt.rcParams.update({'font.size': 12, 'axes.labelsize': 14,
                     'axes.titlesize': 12, 'legend.fontsize': 11,
                     'pdf.fonttype': 42, 'savefig.bbox': 'tight'})
KC = (1 + np.sqrt(33)) / 16

def sf(mu, k, geometry):
    if geometry not in ('sphere', 'cylinder'):
        raise ValueError('geometry must be sphere or cylinder.')
    if np.any(~np.isfinite(np.asarray(mu))) or np.any(np.asarray(mu) <= 0) or not np.isfinite(k) or not 0 <= k <= 1:
        raise ValueError('Require positive permeability and 0 <= k <= 1.')
    g = 2 / 9 * (1-k**3) if geometry == 'sphere' else (1-k**2) / 4
    return 1 + g * (np.asarray(mu)-1)**2 / np.asarray(mu)

def coefficients(mu, a, b, h0, geometry):
    if not np.all(np.isfinite([mu, a, b, h0])) or mu <= 0 or not 0 < a < b or h0 <= 0:
        raise ValueError('Require mu > 0, 0 < a < b and H0 > 0.')
    if geometry == 'sphere':
        den = (mu+2)*(2*mu+1)-2*(a/b)**3*(mu-1)**2
        hi = 9*mu*h0/den
        aa = 3*(2*mu+1)*h0/den
        dd = 3*(mu-1)*a**3*h0/den
        cc = b**3*(mu-1)*(2*mu+1)*(1-(a/b)**3)*h0/den
        p = 2
    elif geometry == 'cylinder':
        den = (mu+1)**2*b**2-(mu-1)**2*a**2
        hi = 4*mu*b**2*h0/den
        aa = 2*(mu+1)*b**2*h0/den
        dd = 2*(mu-1)*a**2*b**2*h0/den
        cc = b**2*(mu**2-1)*(b**2-a**2)*h0/den
        p = 1
    else:
        raise ValueError('Unknown geometry.')
    return hi, aa, dd, cc, p

def fields(mu, a, b, h0, geometry, x, y):
    hi, aa, dd, cc, p = coefficients(mu,a,b,h0,geometry)
    rr = np.hypot(x,y)
    th = np.arctan2(y,x)
    hx,hy = np.zeros_like(x),np.zeros_like(y)
    cavity,wall,exterior = rr<a,(rr>=a)&(rr<=b),rr>b
    hx[cavity] = hi
    for mask, outside in [(wall,False),(exterior,True)]:
        rad,angle = rr[mask],th[mask]
        if outside:
            hr = (h0+p*cc/rad**(p+1))*np.cos(angle)
            ht = -(h0-cc/rad**(p+1))*np.sin(angle)
        else:
            hr = (aa-p*dd/rad**(p+1))*np.cos(angle)
            ht = -(aa+dd/rad**(p+1))*np.sin(angle)
        hx[mask] = hr*np.cos(angle)-ht*np.sin(angle)
        hy[mask] = hr*np.sin(angle)+ht*np.cos(angle)
    mur = np.where(wall,mu,1.)
    # B/B0 = mur H/H0, so mu0 cancels.
    bn = mur*np.hypot(hx,hy)/h0
    return hx,hy,bn,cavity

def validate():
    residuals, sf_residuals = [],[]
    for mu in [1.,2.,10.,1000.,1.e5]:
        for k in [.3,.5,.8,.99]:
            for geom in ['sphere','cylinder']:
                a,b,h0=k,1.,2.3
                hi,aa,dd,cc,p=coefficients(mu,a,b,h0,geom)
                sf_residuals.append(abs(h0/hi-sf(mu,k,geom))/sf(mu,k,geom))
                for rad in [a,b]:
                    hn=aa-p*dd/rad**(p+1)
                    ht=-(aa+dd/rad**(p+1))
                    targetn=hi if rad==a else h0+p*cc/rad**(p+1)
                    targett=-hi if rad==a else -(h0-cc/rad**(p+1))
                    residuals += [abs(mu*hn-targetn)/h0,abs(ht-targett)/h0]
                hi2=coefficients(mu,a,b,2*h0,geom)[0]
                assert np.isclose(hi2,2*hi,rtol=1e-13)
    for mu in [2,10,1000]:
        assert np.isclose(sf(mu,KC,'sphere'),sf(mu,KC,'cylinder'),rtol=1e-13)
        assert sf(mu,.3,'sphere') < sf(mu,.3,'cylinder')
        assert sf(mu,.8,'sphere') > sf(mu,.8,'cylinder')
    for geom in ['sphere','cylinder']:
        assert sf(1,.5,geom)==1
        assert sf(1000,1,geom)==1
    result = {'max_interface_residual_normalized_by_H0':max(residuals),
              'max_relative_SF_consistency_error':max(sf_residuals),
              'crossing_k':KC,
              'tested_mu':[1,2,10,1000,100000],
              'tested_k':[.3,.5,.8,.99],
              'interface_tolerance':1e-10,
              'checks':'interfaces, SF equivalence, crossing, ordering, limits, H0 scaling'}
    assert max(residuals)<1e-10
    assert max(sf_residuals)<1e-10
    (OUT/'validation_revision.json').write_text(json.dumps(result,indent=2)+'\n')
    return result

def geometry_figure():
    fig,axs=plt.subplots(1,2,figsize=(7.2,3.2),layout='constrained')
    for ax,geom in zip(axs,['sphere','cylinder']):
        ax.add_patch(Circle((0,0),1,fc='#d7e5ee',ec='k',lw=1.2))
        ax.add_patch(Circle((0,0),.5,fc='white',ec='k',lw=1.2))
        for end in [(1.35,0),(0,1.3)]:
            ax.annotate('',xy=end,xytext=(-end[0],-end[1]),arrowprops={'arrowstyle':'->','lw':.8})
        ax.text(1.3,-.12,'x');ax.text(.05,1.2,'y')
        ax.annotate('',xy=(1.3,1.15),xytext=(.55,1.15),arrowprops={'arrowstyle':'->','lw':1.5})
        ax.text(.75,1.28,r'$\mathbf{H}_0$')
        ax.annotate('',xy=(.5/np.sqrt(2),-.5/np.sqrt(2)),xytext=(0,0),arrowprops={'arrowstyle':'->','color':'#0868ac'})
        ax.annotate('',xy=(-1/np.sqrt(2),-1/np.sqrt(2)),xytext=(0,0),arrowprops={'arrowstyle':'->','color':'#0868ac'})
        ax.text(.2,-.35,'a',color='#0868ac');ax.text(-.48,-.3,'b',color='#0868ac')
        ax.annotate('',xy=(.75*np.cos(.6),.75*np.sin(.6)),xytext=(0,0),arrowprops={'arrowstyle':'->','color':'#087f23'})
        ax.add_patch(Arc((0,0),.5,.5,theta1=0,theta2=np.degrees(.6),lw=.8))
        ax.text(.31,.08,r'$\theta$')
        ax.text(.6,.46,'r' if geom=='sphere' else r'$\rho$',color='#087f23')
        ax.text(-.82,.35,r'$\mu_0\mu_r$',fontsize=9)
        ax.text(-.28,.18,r'$\mu_0$',fontsize=9)
        ax.set(xlim=(-1.4,1.4),ylim=(-1.18,1.48),aspect='equal')
        ax.axis('off')
    axs[0].set_title('(a) Sphere: meridional section')
    axs[1].set_title('(b) Cylinder: transverse section')
    fig.savefig(OUT/'geometry.pdf');plt.close(fig)

def comparison_figure():
    fig,axs=plt.subplots(1,2,figsize=(7.2,3.1),layout='constrained')
    mu=np.geomspace(1,50000,600)
    for geom,col in [('sphere','#2166ac'),('cylinder','#b2182b')]:
        axs[0].loglog(mu,sf(mu,.5,geom),color=col,label=geom.capitalize(),lw=1.8)
        g=2/9*(1-.5**3) if geom=='sphere' else (1-.5**2)/4
        axs[0].loglog(mu[mu>10],g*mu[mu>10],color=col,ls='--',lw=1)
    axs[0].axhline(1,color='.5',ls=':',lw=1)
    axs[0].set(xlabel=r'$\mu_r$',ylabel='Shielding factor',title='(a) k = 0.5')
    axs[0].legend(loc='upper left');axs[0].grid(alpha=.2)
    kk=np.linspace(.001,1,1500)
    for m,col in [(10,'#1b9e77'),(100,'#d95f02'),(1000,'#7570b3')]:
        ratio=np.array([sf(m,k,'sphere')/sf(m,k,'cylinder') for k in kk])
        axs[1].plot(kk,ratio,color=col,label=rf'$\mu_r={m}$',lw=1.5)
    axs[1].plot(kk,8/9*(1+kk+kk**2)/(1+kk),'k--',lw=1,label='Asymptotic limit')
    axs[1].axhline(1,color='.5',ls=':',lw=1)
    axs[1].axvline(KC,color='.5',ls=':',lw=1)
    axs[1].text(KC+.02,.91,r'$k_c$',fontsize=10)
    axs[1].set(xlim=(0,1),ylim=(.87,1.36),xlabel=r'$k=a/b$',
               ylabel=r'$SF_{\rm sph}/SF_{\rm cyl}$',title='(b) Geometry comparison')
    axs[1].legend(loc='upper left',fontsize=11);axs[1].grid(alpha=.2)
    fig.savefig(OUT/'shielding_comparison.pdf');plt.close(fig)

def field_figure():
    fig,axs=plt.subplots(1,2,figsize=(7.2,3.3),layout='constrained')
    axis=np.linspace(-2,2,501);x,y=np.meshgrid(axis,axis)
    norm=Normalize(-2.5,.7)
    for ax,geom in zip(axs,['sphere','cylinder']):
        hx,hy,bn,cav=fields(1000,.5,1,1,geom,x,y)
        im=ax.pcolormesh(x,y,np.log10(np.maximum(bn,1e-30)),cmap='viridis',norm=norm,shading='auto',rasterized=True)
        ax.streamplot(axis,axis,np.ma.array(hx,mask=cav),np.ma.array(hy,mask=cav),density=.65,color='white',linewidth=.55,arrowsize=.6)
        for yy in [-.26,0,.26]:
            half=np.sqrt(.5**2-yy**2)*.86
            ax.plot([-half,half],[yy,yy],color='white',lw=.8)
            ax.annotate('',xy=(.11,yy),xytext=(-.06,yy),arrowprops={'arrowstyle':'->','color':'white','lw':.8})
        for rad in [.5,1]:ax.add_patch(Circle((0,0),rad,fill=False,ec='k',lw=.9))
        ax.set(xlabel='x/b',ylabel='y/b',aspect='equal',xlim=(-2,2),ylim=(-2,2))
        ax.set_title(('(a) Sphere' if geom=='sphere' else '(b) Cylinder')+f'\nSF = {sf(1000,.5,geom):.3f}')
    cb=fig.colorbar(im,ax=axs,shrink=.9,pad=.025)
    cb.set_label(r'$\log_{10}(|\mathbf{B}|/B_0)$')
    fig.savefig(OUT/'field_maps.pdf',dpi=300);plt.close(fig)

if __name__=='__main__':
    print(json.dumps(validate(),indent=2))
    geometry_figure();comparison_figure();field_figure()
