"""Compatibility imports for HA components; new cases use support.ha."""

from ..support.ha import HA_DUAL_MASTER_ENV as HA_DUAL_MASTER_ENV
from ..support.ha import HA_TRACE_IL as HA_TRACE_IL
from ..support.ha import HA_TRACE_OL as HA_TRACE_OL
from ..support.ha import HA_TRACE_ROWS as HA_TRACE_ROWS
from ..support.ha import HA_TRACE_SPACING_MS as HA_TRACE_SPACING_MS
from ..support.ha import HaRows as HaRows
from ..support.ha import HaTrafficRunner as HaTrafficRunner
from ..support.ha import dual_spec_for_layout as dual_spec_for_layout
from ..support.ha import ha_dual_enabled as ha_dual_enabled
from ..support.ha import ha_gate as ha_gate
from ..support.ha import instance_alive_full as instance_alive_full
from ..support.ha import instance_ops as instance_ops
from ..support.ha import recovery_rate as recovery_rate
from ..support.ha import restore_masters as restore_masters
from ..support.ha import row_ts_ms as row_ts_ms
from ..support.ha import rows_between as rows_between
from ..support.ha import tier1_dual_spec as tier1_dual_spec
from ..support.ha import write_ha_trace as write_ha_trace
