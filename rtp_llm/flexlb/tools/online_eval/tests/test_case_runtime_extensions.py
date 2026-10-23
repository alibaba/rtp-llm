"""Client record validation and report rendering retain their public contracts."""
import time
from unittest.mock import patch
import pytest
from runtime.deadline import Deadline


def test_client_record_decoder_does_not_relabel_invalid_or_stale_rows():
    from case_api.client_records import ClientRecordAdapter, read_records
    adapter = ClientRecordAdapter(lambda: [{'raw': 1}], lambda row: {
        'client_record_schema_version': 1, 'env_epoch': 2, 'wire_request_id': row['raw'], 'attempt': 1})
    deadline = Deadline(time.monotonic() + 5)
    assert read_records(adapter, deadline, 2)[0]['wire_request_id'] == 1
    with pytest.raises(ValueError, match='stale'):
        read_records(adapter, deadline, 3)


def test_selected_report_receives_the_live_render_budget(tmp_path):
    from reporting.workload import write_views
    from case_api.registry import ReportView
    deadline = Deadline(time.monotonic() + 5)
    seen = []
    def render(directory, analysis, presentation, budget):
        budget.check()
        seen.append(budget)
        return directory / 'report.html'
    capability = ReportView(lambda *args: None, lambda *args: None, deadline_renderer=render)
    with patch('reporting.workload.view_capabilities', return_value={'trial.yaml': capability}), \
         patch('reporting.workload.view', return_value=object()):
        assert write_views(tmp_path, {}, ['trial.yaml'], deadline=deadline)['trial.yaml'] == tmp_path / 'report.html'
    assert seen == [deadline]
