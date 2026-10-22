"""Perceptual run shades; metric hue is constant within each ramp.

Oklab matrices: https://bottosson.github.io/posts/oklab/ (public domain).
"""
import math
import json
import colorsys
from reporting.catalog import series_color
from reporting.css_colors import normalize_color
from reporting.report_model import MeasurementIdentity


def to_oklab(color):
    value = normalize_color(color).lstrip('#')
    if len(value) not in (6, 8):
        raise ValueError('comparison colors must be six or eight digit hex')
    rgb = [int(value[k:k+2], 16)/255 for k in (0, 2, 4)]
    r, g, b = [v/12.92 if v <= .04045 else ((v+.055)/1.055)**2.4 for v in rgb]
    l = (.4122214708*r + .5363325363*g + .0514459929*b)**(1/3)
    m = (.2119034982*r + .6806995451*g + .1073969566*b)**(1/3)
    s = (.0883024619*r + .2817188376*g + .6299787005*b)**(1/3)
    return (.2104542553*l + .7936177850*m - .0040720468*s,
            1.9779984951*l - 2.4285922050*m + .4505937099*s,
            .0259040371*l + .7827717662*m - .8086757660*s)


def _linear_rgb(lightness, chroma, hue):
    a, b = chroma*math.cos(hue), chroma*math.sin(hue)
    l = (lightness + .3963377774*a + .2158037573*b)**3
    m = (lightness - .1055613458*a - .0638541728*b)**3
    s = (lightness - .0894841775*a - 1.2914855480*b)**3
    return (4.0767416621*l - 3.3077115913*m + .2309699292*s,
            -1.2684380046*l + 2.6097574011*m - .3413193965*s,
            -.0041960863*l - .7034186147*m + 1.7076147010*s)


def run_color(color, index, count):
    """Equal Oklab distances, softer dark end, and a 3:1 white contrast floor."""
    _, a, b = to_oklab(color)
    hue = math.atan2(b, a)
    chroma = min(.16, max(.10, math.hypot(a, b))) if math.hypot(a, b) > .01 else 0
    lower, upper = .54, .69
    # Fit the whole ramp together: per-color clipping would destroy equal spacing.
    levels = [i/32 for i in range(33)]
    def rgb(t):
        return _linear_rgb(lower+(upper-lower)*t, chroma*(1-.45*t), hue)
    while any(min(rgb(t)) < 0 or max(rgb(t)) > 1 for t in levels):
        chroma *= .98
    while any(sum(v*w for v, w in zip(rgb(t), (.2126, .7152, .0722))) > .298 for t in levels):
        upper -= .001
    r, g, b = rgb(index/max(1, count-1))
    encoded = [12.92*v if v <= .0031308 else 1.055*v**(1/2.4)-.055 for v in (r, g, b)]
    return '#' + ''.join(f'{round(v*255):02x}' for v in encoded)


# Hue identifies the metric; lightness and sparse point shapes identify runs.
RUN_MARKERS = ('circle', 'rect', 'triangle', 'rectRot', 'star', 'cross', 'crossRot')


def metric_colors(panels):
    """Keep existing metric hues where possible; separate near hues within a plot."""
    registry, colors = {}, {}
    for panel in panels:
        used = []
        for n, curve in enumerate(panel.get('series', [])):
            identity = json.dumps(MeasurementIdentity.read(curve, panel.get('unit')).signature(), sort_keys=True)
            if identity == 'null':
                identity = (panel['id'], n)
            color = registry.get(identity) or curve.get('color') or series_color(curve.get('tone'), n)
            value = normalize_color(color).lstrip('#')
            hue, lightness, saturation = colorsys.rgb_to_hls(*(int(value[k:k+2], 16)/255 for k in (0, 2, 4)))
            distance = lambda h: min((min(abs(h-u), 1-abs(h-u)) for u in used), default=1)
            if identity not in registry and saturation >= .10 and distance(hue) < .07:
                hue = max(((hue + k/24) % 1 for k in range(24)), key=distance)
                color = '#' + ''.join(f'{round(v*255):02x}' for v in colorsys.hls_to_rgb(hue, lightness, saturation))
            registry.setdefault(identity, color)
            colors[panel['id'], n] = color
            if saturation >= .10:
                used.append(hue)
    return colors



def curve_style(metric_color, run_index, run_count):
    return dict(color=run_color(metric_color, run_index, run_count), dash=[],
                point_style=RUN_MARKERS[run_index % len(RUN_MARKERS)], marker_count=12)
