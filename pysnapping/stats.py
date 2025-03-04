from dataclasses import dataclass, field
from enum import Enum
import typing
from collections import Counter
import logging

from pysnapping import SnappingError, SnappingMethod

if typing.TYPE_CHECKING:
    from pysnapping.snap import SnappedTripPoints, TrajectoryTrip


logger = logging.getLogger(__name__)


class AggregatedSnappingMethod(Enum):
    failed = "failed"
    all_trusted = "all_trusted"
    all_routed = "all_routed"
    all_fallback = "all_fallback"
    mixed = "mixed"
    total = "total"


@dataclass
class SnappingStats:
    method_counters: list[Counter[SnappingMethod]] = field(default_factory=list)
    aggregated_counter: Counter[AggregatedSnappingMethod] = field(
        default_factory=Counter
    )

    def reset(self) -> None:
        self.method_counters.clear()
        self.aggregated_counter.clear()

    def process_result(
        self, result: "SnappedTripPoints"
    ) -> tuple[Counter[SnappingMethod], AggregatedSnappingMethod]:
        counter = Counter(result.methods)
        self.method_counters.append(counter)
        agg_method = self.aggregate_methods(counter)
        self.aggregated_counter[agg_method] += 1
        self.aggregated_counter[AggregatedSnappingMethod.total] += 1
        return counter, agg_method

    def aggregate_methods(
        self, counter: Counter[SnappingMethod]
    ) -> AggregatedSnappingMethod:
        n_total = sum(v for v in counter.values())
        if counter[SnappingMethod.trusted] == n_total:
            return AggregatedSnappingMethod.all_trusted
        elif counter[SnappingMethod.routed] == n_total:
            return AggregatedSnappingMethod.all_routed
        elif counter[SnappingMethod.fallback] == n_total:
            return AggregatedSnappingMethod.all_fallback
        else:
            return AggregatedSnappingMethod.mixed

    def process_failure(self) -> None:
        agg_method = AggregatedSnappingMethod.failed
        self.aggregated_counter[agg_method] += 1
        self.aggregated_counter[AggregatedSnappingMethod.total] += 1
        logger.debug("snapping failed")

    @typing.overload
    def snap_trip_points(
        self,
        ttrip: "TrajectoryTrip",
        *args,
        log_failed: bool = False,
        raise_failed: typing.Literal[True] = ...,
        **kwargs,
    ) -> "SnappedTripPoints": ...

    @typing.overload
    def snap_trip_points(
        self,
        ttrip: "TrajectoryTrip",
        *args,
        log_failed: bool = False,
        raise_failed: typing.Literal[False],
        **kwargs,
    ) -> typing.Optional["SnappedTripPoints"]: ...

    def snap_trip_points(
        self,
        ttrip: "TrajectoryTrip",
        *args,
        log_failed: bool = False,
        raise_failed: bool = True,
        **kwargs,
    ) -> typing.Optional["SnappedTripPoints"]:
        """Wrapper for `TrajectoryTrip.snap_trip_points` that updates the statistics.

        Snapping statistics and failures are aggregated and logged at level DEBUG. Apart
        from that, by default, snapping errors are raised and not logged. By setting
        `raise_failed` and/or `log_failed` you can choose whether to raise and/or log
        failures at level ERROR (with traceback).

        If you need more custom logging, you should either raise, catch and log or not
        use this wrapper and use `process_result` and `process_failure` instead for even
        more control.
        """
        try:
            result = ttrip.snap_trip_points(*args, **kwargs)
        except SnappingError as error:
            self.process_failure()
            (logger.exception if log_failed else logger.debug)(
                "snapping %r failed: %s",
                ttrip,
                error,
            )
            if raise_failed:
                raise
            else:
                return None
        else:
            counter, agg_method = self.process_result(result)
            logger.debug(
                "snapping %r succeeded with methods %s aggregated to %s",
                ttrip,
                {k.value: v for k, v in counter.items()},
                agg_method.value,
            )
            return result

    def log_aggregated(self, prefix: str = "") -> None:
        logger.info(
            "%s%s",
            prefix,
            {k.value: v for k, v in self.aggregated_counter.items()},
        )
