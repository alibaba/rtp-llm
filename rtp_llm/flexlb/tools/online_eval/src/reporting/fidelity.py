"""Input-distribution diagnostics projected through shared report components."""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

from artifacts.json_io import write_json
from reporting.core import details, table, render as render_spec, _atomic
from reporting.catalog import ACCENT_COLORS

METHOD_LABELS = {'real': '真实输入', 'independent': '独立采样', 'joint': '联合采样'}
METHOD_COLORS = {'real': 'blue', 'independent': 'amber', 'joint': 'green'}


@dataclass(frozen=True)
class FidelityMetrics:
    length_ks: float
    depth_ks: float
    joint_tv: float


@dataclass(frozen=True)
class FidelityMethod:
    name: str
    status: str
    metrics: FidelityMetrics | None
    summary: dict | None
    length_grade: str
    cache_grade: str

    @classmethod
    def read(cls, name, data):
        if data['status'] != 'MEASURED':
            return cls(name, data['status'], None, None, 'UNAVAILABLE', 'UNASSESSED')
        metrics = FidelityMetrics(**data['metrics'])
        return cls(name, data['status'], metrics, data['summary'],
                   data['grades']['length'], data['grades']['cache'])


@dataclass(frozen=True)
class FidelityCapture:
    name: str
    sha256: str
    interpretation: str
    real: dict
    methods: tuple[FidelityMethod, ...]


@dataclass(frozen=True)
class FidelityResult:
    profile: str
    profile_sha256: str
    identity_status: str
    mode: str
    seed: int
    held_out_validated: bool | None
    limitations: tuple[str, ...]
    thresholds: dict
    bins: dict
    captures: tuple[FidelityCapture, ...]

    @classmethod
    def read(cls, report):
        if report['traffic_fidelity_schema_version'] != 1:
            raise ValueError('unsupported traffic fidelity schema')
        return cls(report['profile'], report['profile_sha256'], report['identity']['status'],
                   report['mode'], report['seed'], report['held_out_validated'],
                   tuple(report['limitations']), report['thresholds'], report['bins'],
                   tuple(FidelityCapture(row['capture'], row['capture_sha256'], row['interpretation'],
                         row['real'], tuple(FidelityMethod.read(name, method)
                         for name, method in row['methods'].items())) for row in report['rows']))

    def rows(self):
        for capture in self.captures:
            for method in capture.methods:
                values = ([method.metrics.length_ks, method.metrics.depth_ks, method.metrics.joint_tv]
                          if method.metrics is not None else [None, None, None])
                yield [capture.name, METHOD_LABELS[method.name], *values, method.length_grade, method.cache_grade]


def markdown(report):
    result = FidelityResult.read(report)
    lines = ['# 合成流量保真度', '', f'画像：`{result.profile}`',
             f'画像 SHA：`{result.profile_sha256}`', f'身份核验：{result.identity_status}', '',
             '输入分布诊断不代表真实缓存命中，也不代替运行门禁。', '',
             '| 捕获 | 方法 | length KS | depth KS | joint TV | 长度分布 | cache 使用 |',
             '|---|---|---:|---:|---:|---|---|']
    for row in result.rows():
        lines.append('| ' + ' | '.join('—' if value is None else f'{value:.4f}' if isinstance(value, float)
                                       else str(value).replace('|', '\\|') for value in row) + ' |')
    lines += ['', f'held_out_validated：{result.held_out_validated}', *result.limitations]
    return '\n'.join(lines) + '\n'


def _ecdf(summary, key):
    total = summary['requests']
    cumulative = 0
    points = []
    for value, count in summary[key]:
        points.append(dict(x=value, y=cumulative / total))
        cumulative += count
        points.append(dict(x=value, y=cumulative / total))
    return points


def build_spec(report):
    result = FidelityResult.read(report)
    panels = []
    sections = [table('分布诊断', ['捕获', '方法', 'length KS', 'depth KS', 'joint TV', '长度分布', 'cache 使用'], result.rows()),
                details('诊断口径', dict(identity=result.identity_status, mode=result.mode, seed=result.seed,
                    thresholds=result.thresholds, held_out_validated=result.held_out_validated,
                    limitations=result.limitations,
                    scope='无限历史输入共享不等同有限缓存命中；时间复用、路由和输出关联需单独验证。'))]
    if not result.captures:
        sections.append(details('UNASSESSED', '无对照捕获，不能判断输入保真度。', opened=True))
    for index, capture in enumerate(result.captures):
        summaries = [('real', capture.real)] + [(method.name, method.summary) for method in capture.methods
                                                if method.summary is not None]
        for key, title, unit in [('length_counts', '输入长度 ECDF', 'tokens'), ('depth_counts', '共享深度 ECDF', 'blocks')]:
            panels.append(dict(id=f'{index}.{key}', title=capture.name + ' · ' + title, type='line', timeX=False,
                axes={'y': {'title': '累计请求占比'}}, unit='ratio', caption=capture.interpretation,
                series=[dict(name=METHOD_LABELS[name], unit='ratio', axis='y',
                             color=ACCENT_COLORS[METHOD_COLORS[name]], points=_ecdf(summary, key),
                             description='横轴单位：' + unit) for name, summary in summaries]))
        bins = result.bins
        columns = ['shared blocks / log₂(tokens)'] + [
            f"{bins['length_log2_min'] + i * (bins['length_log2_max'] - bins['length_log2_min']) / bins['length_bins']:.2f}"
            for i in range(bins['length_bins'])]
        for name, summary in summaries:
            grid = summary['joint_counts']
            rows = [[f"{i * bins['depth_max_blocks'] / bins['depth_bins']:.0f}"] +
                    [round(grid[j][i] / summary['requests'], 6) for j in range(bins['length_bins'])]
                    for i in reversed(range(bins['depth_bins']))]
            sections.append(table(capture.name + ' · ' + METHOD_LABELS[name] + ' · 联合密度（请求占比）',
                                  columns, rows, opened=False))
    sections.append(details('冻结分析', report))
    return dict(title='合成流量保真度', subtitle='展示冻结的输入分布诊断；不重判运行门禁。',
                panels=panels, sections=sections, kpis=[])


def render(report):
    return render_spec(build_spec(report))


def write(report, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    write_json(directory / 'fidelity.json', report, indent=2)
    _atomic(directory / 'fidelity.md', markdown(report))
    _atomic(directory / 'fidelity.html', render(report))
