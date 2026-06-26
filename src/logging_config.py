"""
logging_config.py — 프로젝트 공용 로거 설정.

[왜 print가 아니라 logging인가? — 면접 포인트]
- print는 켜고 끌 수 없다. 한번 넣으면 운영에서도 콘솔을 더럽힌다.
- logging은 '레벨'(DEBUG/INFO/WARNING...)로 무엇을 흘릴지 제어할 수 있다.
  개발 땐 DEBUG로 분류 근거(reasoning)까지 보고, 운영 땐 INFO 이상만 남긴다.
- 핸들러/포매터를 한 곳에서 정하면 출력 형식이 전 모듈에서 일관된다.

[관찰가능성(Observability) 관점]
의도 분류는 이 파이프라인에서 가장 자주 틀리고, 틀리면 뒤가 다 무의미해지는
지점이다(그래서 우리는 reasoning 필드를 일부러 받기로 했다). 그 근거를 로그로
남겨두면, 분류가 틀렸을 때 '왜 그렇게 판단했는지'를 사후에 추적할 수 있다.
"""

from __future__ import annotations

import logging
import os


def get_logger(name: str) -> logging.Logger:
    """모듈별 로거를 반환한다. 핸들러 중복 부착을 방지한다.

    로그 레벨은 환경변수 LOG_LEVEL로 제어한다(기본 INFO).
    개발 중 분류 근거까지 보고 싶으면 LOG_LEVEL=DEBUG 로 실행한다.
    """
    logger = logging.getLogger(name)

    # 같은 로거에 핸들러가 여러 번 붙는 것을 막는다(테스트에서 모듈이 반복 import될 때).
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("[%(levelname)s] %(name)s: %(message)s")
        )
        logger.addHandler(handler)

    level_name = os.getenv("LOG_LEVEL", "INFO").upper()
    logger.setLevel(getattr(logging, level_name, logging.INFO))
    # 루트 로거로의 전파를 막아 메시지가 중복 출력되지 않게 한다.
    logger.propagate = False
    return logger
