#!/usr/bin/env python3
"""Photo-guided SFD law illustration; no fitted shape-recovery claim."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import shutil

import numpy as np
from PIL import Image
from scipy.interpolate import PchipInterpolator
from scipy.ndimage import gaussian_filter
from skimage.draw import polygon
from skimage.filters import gaussian
from skimage.measure import find_contours
from pypdf import PdfReader, PdfWriter

import generate_sfd_shape_gallery as gallery
from flowdistribution import SFD, TubeComponent

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT/"outputs/elvis_reference_20261022/author_reference.png"
SELECTION = ROOT/"outputs/elvis_reference_20261022/contour_selection.json"
OLD_GALLERY = ROOT/"output/pdf/SHAPE_GALLERY_20261022/sfd_shape_gallery.pdf"


def to_coordinates(pixels):
    pixels = np.asarray(pixels, dtype=float)
    return np.column_stack([(pixels[:, 0]-190.5)/110, (315-pixels[:, 1])/110])


def smooth_path(points):
    points = np.asarray(points, dtype=float)
    distance = np.r_[0, np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
    return PchipInterpolator(distance, points, axis=0)(np.linspace(0, distance[-1], 400))


def selected_contours(image_path=REFERENCE, selection_path=SELECTION):
    config = json.loads(Path(selection_path).read_text())
    with Image.open(image_path) as image:
        if image.size != (config["reference_width"], config["reference_height"]):
            raise ValueError("Reference dimensions differ from the frozen annotation.")
        values = np.asarray(image.convert("L"), dtype=float)/255
    values = gaussian(values, sigma=config["grayscale_smoothing_sigma_px"], preserve_range=True)
    mask = np.zeros(values.shape, dtype=bool)
    vertices = np.array(config["foreground_polygon"])
    rows, cols = polygon(vertices[:, 1], vertices[:, 0], shape=mask.shape)
    mask[rows, cols] = True
    for exclusion in config.get("automatic_exclusion_polygons", []):
        vertices = np.array(exclusion)
        rows, cols = polygon(vertices[:, 1], vertices[:, 0], shape=mask.shape)
        mask[rows, cols] = False
    paths = []
    for level in config["contour_levels"]:
        for contour in find_contours(values, level, mask=mask):
            pixel_path = contour[:, ::-1]
            length = np.linalg.norm(np.diff(pixel_path, axis=0), axis=1).sum()
            if length >= config["minimum_path_length_px"]:
                paths.append((length, level, to_coordinates(pixel_path)))
    paths.sort(key=lambda item: item[0], reverse=True)
    selected = [{"name": f"image_contour_{i:02d}", "path": p, "level": level,
                 "origin": "grayscale isocontour inside annotated foreground ROI"}
                for i, (_, level, p) in enumerate(paths[:config["maximum_auto_paths"]])]
    for item in config["manual_paths"]:
        selected.append({"name": item["name"], "path": to_coordinates(smooth_path(item["points"])),
                         "level": None, "origin": "photo-guided manual pixel annotation"})
    if not selected:
        raise ValueError("No selected contours.")
    return selected, mask, config


def build_law(selected, sigma):
    return SFD([TubeComponent(name=item["name"], kind="tube",
                              weight=np.linalg.norm(np.diff(item["path"], axis=0), axis=1).sum(),
                              curve=gallery.polyline_curve(item["path"]), cov=np.eye(2)*sigma**2)
                for item in selected], name="elvis_selected_photo_contours")


def density_grid(law, xedges, yedges, nodes=4096):
    count = np.zeros((len(xedges)-1, len(yedges)-1))
    coordinate = (np.arange(nodes)+.5)/nodes
    for component in law.components:
        center = component.curve(coordinate)
        count += np.histogram2d(center[:, 0], center[:, 1], bins=(xedges, yedges),
                                weights=np.full(nodes, component.weight/nodes))[0]
    dx, dy = np.diff(xedges)[0], np.diff(yedges)[0]
    sigma = np.sqrt(law.components[0].cov[0, 0])
    return gaussian_filter(count, sigma=(sigma/dx, sigma/dy), mode="constant", truncate=6)/(dx*dy)


def draw(selected, law, samples, density, xedges, yedges, out, pdf):
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    gallery.set_sfd_matplotlib_style(plt, base_size=9, title_size=10, label_size=9, tick_size=8)
    plt.rcParams.update({"text.color": gallery.INK, "axes.labelcolor": gallery.INK})
    fig = plt.figure(figsize=(8, 7.9))
    positions = [[.065,.565,.37,.35],[.55,.565,.34,.35],[.065,.115,.37,.35]]
    axes = [fig.add_axes(p) for p in positions]
    for ax in axes:
        ax.set(aspect="equal", xlim=(xedges[0],xedges[-1]), ylim=(yedges[0],yedges[-1]),
               xlabel=r"$X_1$", ylabel=r"$X_2$")
        gallery.frame(ax)
    for item in selected:
        color = gallery.TEAL if item["name"] == "cable" else gallery.INK
        width = .7 if item["name"] in ("pose_outline", "cable") else .38
        axes[0].plot(*item["path"].T, color=color, lw=width)
    mesh = axes[1].pcolormesh(xedges, yedges, density.T, cmap=gallery.CMAP,
                              shading="flat", rasterized=True)
    colorbar = fig.colorbar(mesh, cax=fig.add_axes([.915,.625,.012,.22]))
    colorbar.set_label("Density", fontsize=8)
    colorbar.ax.tick_params(labelsize=7, width=.5)
    axes[2].scatter(*samples.T, s=.32, color=gallery.TEAL, alpha=.5, linewidths=0,
                    rasterized=True)
    surface = fig.add_axes([.51,.09,.42,.38], projection="3d")
    centers = [(edges[:-1]+edges[1:])/2 for edges in (xedges,yedges)]
    xx, yy = np.meshgrid(*centers, indexing="ij")
    surface.plot_surface(xx, yy, density, cmap=gallery.CMAP, rcount=180, ccount=180,
                          linewidth=0, antialiased=True, rasterized=True)
    surface.set(xlabel=r"$X_1$", ylabel=r"$X_2$", zlabel="Density")
    gallery.clean3(surface, aspect=(1,1,.45), elev=50, azim=-80, zoom=1.0)
    for x,y,title in ((.065,.948,"(a) Photo-guided contours"),
                      (.55,.948,"(b) Gaussian-tube density"),
                      (.065,.498,"(c) Generated observations"),
                      (.55,.498,"(d) Probability-density surface")):
        gallery.heading(fig,x,y,title)
    fig.text(.065,.027,r"18,000 draws; $\Sigma_b=0.014^2 I_2$. Height in (d) is density, not geometric depth.",
             fontsize=8.5,color="#697377")
    with PdfPages(pdf) as book:
        gallery.export(fig,"elvis_sfd_four_panel",out,book)


def generate(source, out, pdf_dir):
    out, pdf_dir = Path(out), Path(pdf_dir)
    if out.exists() or pdf_dir.exists():
        raise FileExistsError("Use fresh output directories; previous candidates are preserved.")
    if not REFERENCE.exists():
        REFERENCE.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(source,REFERENCE)
    elif Path(source).read_bytes()!=REFERENCE.read_bytes():
        raise ValueError("The supplied image differs from the frozen author reference.")
    selected, mask, config = selected_contours()
    out.mkdir(parents=True)
    pdf_dir.mkdir(parents=True)
    contract = {"purpose":"specified image-contour SFD; not fitted recovery",
                "reference":"author-supplied crouching pose with guitar and microphone",
                "historical_date_and_photographer_verified":False,
                "photo_reproduced_in_deliverable":False,"publication_use_cleared":False,
                "contour_extraction":"ROI-selected grayscale isocontours plus declared pose/microphone/cable annotations",
                "sample_size":config["sample_size"],"seed":config["seed"],
                "kernel_sd":config["kernel_sd"],"mass_rule":config["mass_rule"],
                "density_display":"binned midpoint quadrature followed by Gaussian convolution; not fitted KDE",
                "quadrature_nodes_per_curve":4096,"new_fits":0,"manuscript_replaced":False}
    (out/"design_contract.json").write_text(json.dumps(contract,indent=2)+"\n")
    law = build_law(selected,config["kernel_sd"])
    samples, info = law.rvs(config["sample_size"],random_state=config["seed"],return_info=True)
    gallery.save_sample(out/"elvis_observations.csv",samples,info)
    all_points = np.vstack([item["path"] for item in selected])
    lo, hi = all_points.min(0)-.10, all_points.max(0)+.10
    xedges, yedges = [np.linspace(lo[j],hi[j],501) for j in (0,1)]
    density = density_grid(law,xedges,yedges)
    np.savez_compressed(out/"elvis_density.npz",density=density,xedges=xedges,yedges=yedges,mask=mask)
    with (out/"elvis_contours.csv").open("w",newline="") as stream:
        writer=csv.writer(stream)
        writer.writerow(["component","point","X1","X2","origin","level"])
        for item in selected:
            writer.writerows([item["name"],j,*p,item["origin"],item["level"]]
                             for j,p in enumerate(item["path"]))
    standalone=pdf_dir/"elvis_sfd_example.pdf"
    draw(selected,law,samples,density,xedges,yedges,out,standalone)
    combined=pdf_dir/"sfd_shape_gallery_with_elvis.pdf"
    writer=PdfWriter()
    writer.append(str(standalone))
    writer.append(str(OLD_GALLERY))
    writer.add_metadata({"/Title":"SFD Shape Gallery with Author-Selected Elvis Example",
                         "/Subject":"Internal figure selection; source-photo reuse review pending"})
    with combined.open("wb") as stream:
        writer.write(stream)
    report={"components":len(selected),"automatic_contours":len(selected)-len(config["manual_paths"]),
            "annotated_paths":[p["name"] for p in config["manual_paths"]],
            "mixture_mass":float(law.weights.sum()),
            "display_density_integral":float(density.sum()*np.diff(xedges)[0]*np.diff(yedges)[0]),
            "samples_in_display":int(np.sum(np.all((samples>=lo)&(samples<=hi),axis=1))),
            "sample_size":len(samples),"standalone_pages":len(PdfReader(standalone).pages),
            "combined_pages":len(PdfReader(combined).pages),"new_fits":0,"photo_reproduced":False}
    (out/"generation_checks.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report,indent=2))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    parser.add_argument("--pdf-dir",type=Path,required=True)
    args=parser.parse_args()
    generate(args.image.resolve(),args.out.resolve(),args.pdf_dir.resolve())
