# Copyright 2016 Julien Danjou
# Copyright 2016 Joshua Harlow
# Copyright 2013-2014 Ray Holder
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import traceback
import typing

from tenacity import _utils

if typing.TYPE_CHECKING:
    from tenacity import RetryCallState


def before_sleep_nothing(retry_state: "RetryCallState") -> None:
    """Before sleep strategy that does nothing."""


def before_sleep_log(
    logger: _utils.LoggerProtocol,
    log_level: int,
    exc_info: bool = False,
    sec_format: str = "%.3g",
) -> typing.Callable[["RetryCallState"], None]:
    """Before sleep strategy that logs to some logger the attempt."""


    return log_it
