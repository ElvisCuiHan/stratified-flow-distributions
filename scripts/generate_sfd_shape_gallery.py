#!/usr/bin/env python3
"""Auditable histograms, ring density, Fermat section and optional photo contours."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT/"tmp/matplotlib"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
from PIL import Image
from pypdf import PdfReader, PdfWriter
from scipy.ndimage import gaussian_filter
from scipy.special import i0e

from flowdistribution import GaussianComponent, SFD, TubeComponent
from sfd_figure_style import set_sfd_matplotlib_style

INK = "#343A3E"
TEAL = "#267C7C"
PINK = "#B86477"
CY_COLORS = ("#559EA3", "#D59A78", "#8DA977", "#7E9FBF", "#BD869E")
CMAP = LinearSegmentedColormap.from_list("sfd_mint_amber_coral", [
    (0., "#FCFDFB"), (.16, "#D9EDE6"), (.43, "#9CD8CC"),
    (.65, "#54BCA9"), (.82, "#E5D88A"), (1., "#E98B70")])
CLASSICAL = ROOT/"outputs/jmlr_large_scale_execution_v2/classical_distribution_rescue_small/classical_distribution_samples.csv"
SCENARIOS = (("mvnormal_correlated_10d", "Correlated Gaussian"),
             ("skew_mixture_10d", "Skew mixture"), ("banana_2d", "Banana transform"))
CY_U = (0.025, np.pi/2-0.025)
CY_V = (-1., 1.)
SEEDS = {"elvis": 2026102205, "volcano": 2026102206, "fermat": 2026102207}


def frame(ax):
    ax.spines[["top", "right"]].set_visible(False)
    for spine in ax.spines.values():
        spine.set_color("#ABB3B5")
        spine.set_linewidth(.5)
    ax.tick_params(length=2.5, width=.5, pad=3, color="#939C9F")


def clean3(ax, *, aspect=(1, 1, .6), elev=32, azim=-52, zoom=1.05, axes=True):
    ax.view_init(elev=elev, azim=azim)
    ax.set_proj_type("ortho")
    ax.set_box_aspect(aspect, zoom=zoom)
    ax.grid(False)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.fill = False
        axis.pane.set_edgecolor((1, 1, 1, 0))
        axis.line.set_color("#BAC0C2")
        axis.line.set_linewidth(.5)
        axis.labelpad = 1
    ax.tick_params(pad=0, labelsize=7)
    if not axes:
        ax.set_axis_off()


def heading(fig, x, y, text):
    fig.text(x, y, text, fontsize=10, weight="bold", va="bottom")


def export(fig, key, out, book):
    fig.canvas.draw()
    texts=list(fig.texts)
    for ax in fig.axes:
        texts += [ax.title, *ax.texts]
        for name in ("xaxis", "yaxis", "zaxis"):
            axis=getattr(ax,name,None)
            if axis is None:
                continue
            texts.append(axis.label)
            lo,hi=sorted(axis.get_view_interval())
            for tick in axis.get_major_ticks():
                # Auto locators also create off-range ticks that are never drawn.
                if lo-1e-10 <= tick.get_loc() <= hi+1e-10:
                    texts += [tick.label1,tick.label2]
    for artist in texts:
        if not artist.get_visible() or not artist.get_text().strip():
            continue
        bounds = artist.get_window_extent(fig.canvas.get_renderer())
        assert bounds.x0 >= -1 and bounds.y0 >= -1, (key, artist.get_text())
        assert bounds.x1 <= fig.bbox.width+1 and bounds.y1 <= fig.bbox.height+1, (key, artist.get_text())
    fig.savefig(out/(key+".png"), dpi=240)
    book.savefig(fig)
    plt.close(fig)


def save_sample(path, values, info):
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["id", *[f"X{j+1}" for j in range(values.shape[1])], "component", "s"])
        for i, point in enumerate(values):
            writer.writerow([i+1, *point, info["component"][i], info["s"][i]])


def histogram2(values, xedges, yedges):
    counts, _, _ = np.histogram2d(values[:, 0], values[:, 1], bins=(xedges, yedges))
    area = np.diff(xedges)[:, None]*np.diff(yedges)[None, :]
    # Divide by the full sample size, not the count retained in a zoom window.
    return counts, counts/(len(values)*area)


def load_histogram_inputs():
    selected = {scenario: [] for scenario, _ in SCENARIOS}
    with CLASSICAL.open(newline="") as stream:
        for row in csv.DictReader(stream):
            if row["split"] != "train" or row["rep"] != "1" or row["scenario"] not in selected:
                continue
            selected[row["scenario"]].append([row["sample_id"], float(row["x01"]), float(row["x02"])])
    assert all(len(v) == 500 for v in selected.values())
    return selected


def draw_existing_histograms(out, book):
    selected = load_histogram_inputs()
    fig = plt.figure(figsize=(8, 4.5))
    records = []
    for j, (scenario, title) in enumerate(SCENARIOS):
        values = np.array([r[1:] for r in selected[scenario]])
        with (out/("histogram_"+scenario+".csv")).open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["sample_id", "X1", "X2"])
            writer.writerows(selected[scenario])
        edges = [np.linspace(np.floor(values[:, k].min())-.1, np.ceil(values[:, k].max())+.1, 25)
                 for k in (0, 1)]
        counts, density = histogram2(values, *edges)
        assert counts.sum() == len(values)
        np.savez_compressed(out/("histogram_"+scenario+".npz"), counts=counts,
                            density=density, xedges=edges[0], yedges=edges[1])
        ax = fig.add_axes([.06+j*.31, .44, .27, .43], projection="3d")
        xx, yy = np.meshgrid(edges[0][:-1], edges[1][:-1], indexing="ij")
        keep = counts.ravel() > 0
        ax.bar3d(xx.ravel()[keep], yy.ravel()[keep], np.zeros(keep.sum()),
                 np.diff(edges[0])[0]*.94, np.diff(edges[1])[0]*.94, density.ravel()[keep],
                 color=CMAP(density.ravel()[keep]/density.max()), edgecolor="white",
                 linewidth=.12, shade=True, zsort="average")
        ax.set(xlabel=r"$X_1$", ylabel=r"$X_2$", zlabel="Density")
        clean3(ax, aspect=(1, 1, .8), zoom=.95)
        lower = fig.add_axes([.075+j*.332, .13, .24, .20])
        lower.hist(values[:, 0], bins=edges[0], density=True, color=TEAL, alpha=.65,
                   edgecolor="white", linewidth=.5)
        lower.set(xlabel=r"$X_1$", ylabel="Density")
        frame(lower)
        heading(fig, .045+j*.332, .92, f"({chr(97+j)}) {title}")
        records.append({"scenario": scenario, "n": len(values), "split": "train", "rep": 1,
                        "bivariate_bins": [24, 24], "histogram_integral": float(np.sum(density*np.diff(edges[0])[:, None]*np.diff(edges[1])[None, :]))})
    fig.text(.075, .025, "Existing training samples; 500 per scenario. Bar height is empirical bin density.", fontsize=8.5, color="#697377")
    export(fig, "H_existing_histograms", out, book)
    return records


def polyline_curve(points):
    points = np.asarray(points, dtype=float)
    steps = np.linalg.norm(np.diff(points, axis=0), axis=1)
    distance = np.r_[0, np.cumsum(steps)]
    distance /= distance[-1]
    return lambda s: np.column_stack([np.interp(s, distance, points[:, j]) for j in range(points.shape[1])])


def portrait_law(paths):
    lengths = [np.linalg.norm(np.diff(p, axis=0), axis=1).sum() for p in paths]
    return SFD([TubeComponent(name=f"contour_{j:03d}", kind="tube", weight=length,
                             curve=polyline_curve(p), cov=np.eye(2)*.022**2)
                for j, (p, length) in enumerate(zip(paths, lengths))], name="stage_profile_illustration")


def photo_contours(path):
    from skimage.filters import gaussian
    from skimage.measure import find_contours
    with Image.open(path) as original:
        image=original.convert("L")
        image.thumbnail((1000,1000),Image.Resampling.LANCZOS)
        values=np.asarray(image,dtype=float)/255
    values=gaussian(values,sigma=1.7,preserve_range=True)
    height,width=values.shape
    paths=[]
    for level in (.20,.35,.50,.65,.80):
        for contour in find_contours(values,level):
            length=np.linalg.norm(np.diff(contour,axis=0),axis=1).sum()
            if length < .12*height:
                continue
            x=(contour[:,1]-width/2)/(height/3.2)
            y=(height/2-contour[:,0])/(height/3.2)
            paths.append(np.column_stack([x,y]))
    paths.sort(key=lambda p:np.linalg.norm(np.diff(p,axis=0),axis=1).sum(),reverse=True)
    if not paths:
        raise ValueError("No usable image contours; check the exact reference image.")
    return paths[:80]


def grid_density(law, xedges, yedges):
    # Binned quadrature of a specified Gaussian-kernel law; no density fitting.
    count = np.zeros((len(xedges)-1, len(yedges)-1))
    s = (np.arange(2048)+.5)/2048
    for component in law.components:
        centers = component.curve(s)
        count += np.histogram2d(centers[:, 0], centers[:, 1], bins=(xedges, yedges),
                               weights=np.full(len(s), component.weight/len(s)))[0]
    dx, dy = xedges[1]-xedges[0], yedges[1]-yedges[0]
    sigma = np.sqrt(law.components[0].cov[0, 0])
    return gaussian_filter(count, sigma=(sigma/dx, sigma/dy), mode="constant", truncate=5)/(dx*dy)


def draw_portrait(out, book, *, paths=None, source=None):
    if paths is None:
        raise ValueError("The author requested a specific stage photograph; supply its contours.")
    if not source:
        raise ValueError("Image-derived examples require an explicit source credit.")
    law = portrait_law(paths)
    sample, info = law.rvs(18000, random_state=SEEDS["elvis"], return_info=True)
    save_sample(out/"elvis_observations.csv", sample, info)
    points = np.vstack(paths)
    lo, hi = points.min(0)-.18, points.max(0)+.18
    xedges, yedges = np.linspace(lo[0], hi[0], 401), np.linspace(lo[1], hi[1], 401)
    density = grid_density(law, xedges, yedges)
    np.savez_compressed(out/"elvis_density.npz", density=density, xedges=xedges, yedges=yedges)
    with (out/"elvis_curves.csv").open("w", newline="") as stream:
        writer=csv.writer(stream)
        writer.writerow(["component", "point", "X1", "X2"])
        for j,p in enumerate(paths):
            writer.writerows([f"contour_{j:03d}", k, *v] for k,v in enumerate(p))
    fig=plt.figure(figsize=(7.6,4.6))
    left=fig.add_axes([.06,.13,.34,.75])
    for path in paths:
        left.plot(path[:,0],path[:,1],color=INK,lw=.8)
    left.set_aspect("equal")
    left.set(xlim=(lo[0],hi[0]),ylim=(lo[1],hi[1]),xlabel=r"$X_1$",ylabel=r"$X_2$")
    frame(left)
    right=fig.add_axes([.53,.13,.34,.75])
    mesh=right.pcolormesh(xedges,yedges,density.T,cmap=CMAP,shading="flat",rasterized=True)
    right.set_aspect("equal")
    right.set(xlabel=r"$X_1$",ylabel=r"$X_2$")
    frame(right)
    cb=fig.colorbar(mesh,cax=fig.add_axes([.905,.26,.014,.45]))
    cb.set_label("Density",fontsize=8)
    cb.ax.tick_params(labelsize=7,width=.5)
    heading(fig,.06,.94,"(a) Elvis Presley: image contours")
    heading(fig,.53,.94,"(b) Specified Gaussian-tube density")
    fig.text(.06,.025,"Archival image used only to declare contours; no fitted recovery.",fontsize=8.5,color="#697377")
    export(fig,"E_elvis_profile",out,book)
    return {"source":source, "components":len(paths),
            "sample_size":18000,"kernel_sd":.022,"weight_rule":"proportional to declared contour length",
            "density_grid_integral":float(density.sum()*np.diff(xedges)[0]*np.diff(yedges)[0]),
            "density_display":"binned 2048-node-per-contour Gaussian quadrature; not fitted KDE"}


def volcano_law():
    return SFD([TubeComponent(name="rim",kind="tube",weight=.94,
                curve=lambda s: 1.2*np.column_stack([np.cos(2*np.pi*s),np.sin(2*np.pi*s)]),
                cov=np.eye(2)*.16**2),
                GaussianComponent(name="residual",kind="gaussian",weight=.06,
                                  loc=[0,0],cov=np.eye(2)*1.25**2)], name="volcano_ring")


def volcano_pdf(radius):
    r=np.asarray(radius)
    sigma,R=.16,1.2
    ring=np.exp(-.5*((r-R)/sigma)**2)*i0e(r*R/sigma**2)/(2*np.pi*sigma**2)
    ambient=np.exp(-.5*(r/1.25)**2)/(2*np.pi*1.25**2)
    return .94*ring+.06*ambient


def draw_volcano(out,book):
    law=volcano_law()
    sample,info=law.rvs(16000,random_state=SEEDS["volcano"],return_info=True)
    save_sample(out/"volcano_observations.csv",sample,info)
    x=np.linspace(-2.25,2.25,161)
    xx,yy=np.meshgrid(x,x)
    density=volcano_pdf(np.hypot(xx,yy))
    edges=np.linspace(-2.25,2.25,65)
    counts,hist=histogram2(sample,edges,edges)
    np.savez_compressed(out/"volcano_fields.npz",x=x,density=density,counts=counts,
                        histogram=hist,edges=edges)
    fig=plt.figure(figsize=(8,4.6))
    surface=fig.add_axes([.005,.35,.47,.53],projection="3d")
    surface.plot_surface(xx,yy,density,cmap=CMAP,linewidth=0,antialiased=True,
                         rcount=150,ccount=150,rasterized=True)
    surface.contour(xx,yy,density,zdir="z",offset=0,levels=8,colors=TEAL,linewidths=.3,alpha=.45)
    surface.set(xlabel=r"$X_1$",ylabel=r"$X_2$",zlabel="Density")
    clean3(surface,aspect=(1,1,.55),elev=32,zoom=1.1)
    image=fig.add_axes([.59,.48,.25,.37])
    mesh=image.pcolormesh(edges,edges,hist.T,cmap=CMAP,shading="flat",rasterized=True)
    image.set(aspect="equal",xlabel=r"$X_1$",ylabel=r"$X_2$")
    frame(image)
    cb=fig.colorbar(mesh,cax=fig.add_axes([.88,.51,.013,.29]))
    cb.set_label("Bin density",fontsize=8)
    cb.ax.tick_params(labelsize=7)
    radius=np.linalg.norm(sample,axis=1)
    radial=fig.add_axes([.11,.17,.75,.15])
    radial.hist(radius,bins=np.linspace(0,radius.max()+1e-8,85),density=True,color="#82BFB0",
                 edgecolor="white",lw=.4,label="Sample histogram")
    r=np.linspace(0,radius.max()+.05,800)
    radial.plot(r,2*np.pi*r*volcano_pdf(r),color=PINK,lw=1.1,label="Specified radial density")
    radial.set(xlabel=r"$r=\sqrt{X_1^2+X_2^2}$",ylabel="Density")
    radial.legend(frameon=False,fontsize=8,loc="upper right")
    frame(radial)
    heading(fig,.04,.94,"(a) Volcano-shaped probability density")
    heading(fig,.57,.94,"(b) Empirical histogram")
    fig.text(.11,.025,"Vertical height in (a) is probability density, not a third observed coordinate.",fontsize=8.5,color="#697377")
    export(fig,"F_volcano_density",out,book)
    return {"sample_size":16000,"ring_mass":.94,"radius":1.2,"kernel_sd":.16,
            "residual_mass":.06,"residual_sd":1.25,"displayed_2d_count":int(counts.sum()),
            "outside_2d_window":int(len(sample)-counts.sum()),"radial_histogram_all_points":True}


def fermat_coordinates(u,v,k,l):
    w=np.asarray(u)+1j*np.asarray(v)
    z1=np.exp(.4*np.log(np.sin(w)))*np.exp(2j*np.pi*k/5)
    z2=np.exp(.4*np.log(np.cos(w)))*np.exp(2j*np.pi*l/5)
    return z1,z2


def fermat_projection(u,v,k,l):
    z1,z2=fermat_coordinates(u,v,k,l)
    return np.stack([z1.real,z2.real,(z1.imag+z2.imag)/np.sqrt(2)],axis=-1)


def fermat_law():
    components=[]
    v_nodes=np.linspace(CY_V[0],CY_V[1],13)
    for k in range(5):
        for l in range(5):
            for j,v in enumerate(v_nodes):
                curve=lambda s,k=k,l=l,v=v: fermat_projection(CY_U[0]+np.asarray(s)*(CY_U[1]-CY_U[0]),v,k,l)
                components.append(TubeComponent(name=f"patch_{k}_{l}_v{j:02d}",kind="tube",weight=1,
                                  curve=curve,cov=np.eye(3)*.035**2))
    return SFD(components,name="projected_fermat_section_grid")


def draw_fermat(out,book):
    law=fermat_law()
    sample,info=law.rvs(18000,random_state=SEEDS["fermat"],return_info=True)
    save_sample(out/"fermat_observations.csv",sample,info)
    u,v=np.meshgrid(np.linspace(*CY_U,51),np.linspace(*CY_V,45))
    fig=plt.figure(figsize=(8,4.7))
    left=fig.add_axes([.0,.26,.49,.64],projection="3d",computed_zorder=True)
    right=fig.add_axes([.50,.26,.49,.64],projection="3d",computed_zorder=True)
    residual=0.
    surface_records=[]
    for k in range(5):
        for l in range(5):
            z1,z2=fermat_coordinates(u,v,k,l)
            residual=max(residual,float(np.max(np.abs(-1+z1**5+z2**5))))
            points=fermat_projection(u,v,k,l)
            left.plot_surface(points[:,:,0],points[:,:,1],points[:,:,2],color=CY_COLORS[k],
                              rcount=45,ccount=35,linewidth=0,alpha=.84,shade=True,rasterized=True)
            surface_records.append(points.reshape(-1,3))
        mask=np.array([name.startswith(f"patch_{k}_") for name in info["component"]])
        right.scatter(*sample[mask].T,s=.85,color=CY_COLORS[k],alpha=.62,linewidths=0,depthshade=False,rasterized=True)
    surfaces=np.concatenate(surface_records)
    np.savez_compressed(out/"fermat_section.npz",surfaces=surfaces,
                        u=np.linspace(*CY_U,51),v=np.linspace(*CY_V,45))
    for ax in (left,right):
        ax.set(xlim=(-1.65,1.65),ylim=(-1.65,1.65),zlim=(-1.8,1.8))
        clean3(ax,aspect=(1,1,1.1),elev=25,azim=-52,zoom=1.15,axes=False)
    # Here plot axes are X1, X2, X3 rather than the face figure's reordered axes.
    for ax in (left,right):
        ax.text2D(.10,.08,r"$(X_1,X_2,X_3)$",transform=ax.transAxes,fontsize=8,color="#697377")
    heading(fig,.04,.94,"(a) Fermat-quintic section: 3D projection")
    heading(fig,.54,.94,"(b) Observations around projected patches")
    fig.text(.07,.15,r"$[-1:z_1:z_2:0:0],\qquad z_1^5+z_2^5=1$",fontsize=10)
    fig.text(.07,.085,"A two-real-dimensional section, not the full six-real-dimensional Calabi-Yau space.",fontsize=8.5,color="#697377")
    fig.text(.07,.035,"The sampling law is declared in parameter coordinates, not Calabi-Yau volume.",fontsize=8.5,color="#697377")
    export(fig,"G_fermat_section",out,book)
    return {"sample_size":18000,"components":325,"u_range":list(CY_U),"v_range":list(CY_V),
            "v_nodes":13,"kernel_sd":.035,"max_complex_equation_residual":residual,
            "projection":"(Re z1, Re z2, (Im z1+Im z2)/sqrt(2))",
            "geometry_source":"https://arxiv.org/html/2407.06914v2#S3.SS1",
            "volume_measure_claim":False,"ricci_flat_metric_computed":False,
            "latent_sampler":"uniform mixture over 25 fifth-root pairs and 13 v nodes; continuous uniform u"}


def comparison(out):
    names=["H_existing_histograms","E_elvis_profile","F_volcano_density","G_fermat_section"]
    names=[name for name in names if (out/(name+".png")).exists()]
    sheet=Image.new("RGB",(1800,1330),"white")
    for j,name in enumerate(names):
        with Image.open(out/(name+".png")) as image:
            if len(names)==3:
                if j==0:
                    image.thumbnail((1400,650),Image.Resampling.LANCZOS)
                    sheet.paste(image,((1800-image.width)//2,(665-image.height)//2))
                else:
                    image.thumbnail((890,650),Image.Resampling.LANCZOS)
                    sheet.paste(image,(900*(j-1)+(900-image.width)//2,665+(665-image.height)//2))
                continue
            image.thumbnail((890,650),Image.Resampling.LANCZOS)
            sheet.paste(image,(900*(j%2)+(900-image.width)//2,665*(j//2)+(665-image.height)//2))
    sheet.save(out/"shape_gallery_overview.png")


def generate(out,pdf,portrait=None,portrait_credit=None):
    if portrait is not None and not portrait_credit:
        raise ValueError("Image-derived examples require an explicit source credit.")
    if out.exists() or pdf.exists():
        raise FileExistsError("Use fresh output locations; existing galleries are preserved.")
    out.mkdir(parents=True)
    pdf.parent.mkdir(parents=True,exist_ok=True)
    contract={"purpose":"visual-law selection only; not fitted performance",
              "seeds":SEEDS,"sample_sizes":{"volcano":16000,"fermat":18000},
              "elvis_planned_sample_size_if_reference_supplied":18000,
              "existing_histograms":{"scenarios":SCENARIOS,"split":"train","rep":1,"n_each":500,"bins":24},
              "elvis_source":portrait_credit or "exact 1969 crouching-microphone photo pending; no substitute published",
              "reference_photo_used":portrait is not None,
              "volcano":{"R":1.2,"sigma":.16,"ring_mass":.94,"residual_sd":1.25},
              "fermat":{"u_range":list(CY_U),"v_range":list(CY_V),"v_nodes":13,"root_pairs":25,"sigma":.035},
              "new_fits":0,"new_external_research_data":False,"new_holdout_scoring":False,
              "manuscript_replaced":False,"new_performance_claims":False}
    (out/"design_contract.json").write_text(json.dumps(contract,indent=2)+"\n")
    set_sfd_matplotlib_style(plt,base_size=9,title_size=10,label_size=9,tick_size=8)
    plt.rcParams.update({"text.color":INK,"axes.labelcolor":INK})
    temporary=out/"generated_pages.pdf"
    with PdfPages(temporary) as book:
        report={"histograms":draw_existing_histograms(out,book)}
        report["elvis"]=(draw_portrait(out,book,paths=photo_contours(portrait),source=portrait_credit)
                          if portrait is not None else
                          {"status":"awaiting exact author-selected photograph; no substitute included"})
        report["volcano"]=draw_volcano(out,book)
        report["fermat"]=draw_fermat(out,book)
    writer=PdfWriter()
    writer.append(str(temporary))
    original=ROOT/"figures/fig_external_family_density_illustrations.pdf"
    writer.append(str(original))
    writer.append(str(ROOT/"figures/scgtm_sfd/sfd_eval_tube_residual_diagnostics.pdf"))
    writer.add_metadata({"/Title":"SFD: Histograms and Shape Examples","/Author":"",
                         "/Subject":"Author-selection gallery; not new fitted performance"})
    with pdf.open("wb") as stream:
        writer.write(stream)
    report["pages"]=len(PdfReader(pdf).pages)
    report["appended_existing_gallery"]="existing kernel-smoothed sample-density illustrations, not histograms or fitted-SFD results"
    report["appended_existing_residual_diagnostic"]="archived input-level scGTM-SFD residual diagnostic; not independent estimator validation"
    (out/"generation_checks.json").write_text(json.dumps(report,indent=2)+"\n")
    comparison(out)
    print(json.dumps(report,indent=2))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out",type=Path,required=True)
    parser.add_argument("--pdf",type=Path,required=True)
    parser.add_argument("--portrait",type=Path)
    parser.add_argument("--portrait-credit")
    args=parser.parse_args()
    generate(args.out.resolve(),args.pdf.resolve(),args.portrait,args.portrait_credit)
