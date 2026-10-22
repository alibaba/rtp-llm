"""Unavailable resource accounting never becomes zero usage or a business fault."""
from diagnostics.process_resources import process_resources


def test_cgroup_v1_is_explicitly_unavailable_and_process_rss_remains_usable(tmp_path):
    process = tmp_path/'proc/123'
    process.mkdir(parents=True)
    (process/'status').write_text('Name:\tmock\nVmRSS:\t2048 kB\n')
    (process/'cgroup').write_text('7:memory:/docker/a\n6:cpu,cpuacct:/docker/a\n')
    result = process_resources(123,proc_root=tmp_path/'proc',cgroup_root=tmp_path/'cgroups')
    assert result['rss_bytes'] == 2097152
    assert result['cpu_usec'] is None and result['memory_bytes'] is None
    assert result['unavailable'] == 'cgroup v2 accounting unavailable'


def test_cgroup_v2_preserves_counter_units_and_labels_shared_accounting_scope(tmp_path):
    process = tmp_path/'proc/123'
    process.mkdir(parents=True)
    (process/'status').write_text('VmRSS:\t1 kB\n')
    (process/'cgroup').write_text('0::/job\n')
    group=tmp_path/'cgroups/job'
    group.mkdir(parents=True)
    for name,text in {'cpu.stat':'usage_usec 1234\nthrottled_usec 7\n','memory.current':'4096',
                      'cpu.max':'100000 100000','memory.max':'max'}.items():
        (group/name).write_text(text)
    result=process_resources(123,proc_root=tmp_path/'proc',cgroup_root=tmp_path/'cgroups')
    assert result['cpu_usec']==1234 and result['throttled_usec']==7
    assert result['memory_bytes']==4096 and result['rss_bytes']==1024
    assert result['accounting_scope']=='cgroup_v2' and result['cgroup_path']=='/job'
    assert result['unavailable'] is None
    (group/'memory.current').unlink()
    partial=process_resources(123,proc_root=tmp_path/'proc',cgroup_root=tmp_path/'cgroups')
    assert partial['rss_bytes']==1024 and partial['memory_bytes'] is None
    assert partial['unavailable'] is not None


def test_gone_process_does_not_fabricate_resource_values(tmp_path):
    result=process_resources(123,proc_root=tmp_path)
    assert result['rss_bytes'] is None and result['cpu_usec'] is None
    assert result['unavailable'] is not None
