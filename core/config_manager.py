"""설정 로드/수정 관리자 — LLM 연동으로 trading-params.yaml 자동 조정."""

import copy
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml
from loguru import logger


CONFIG_DIR = Path(__file__).parent.parent / "config"


class ConfigManager:
    """YAML 설정 파일을 로드하고, LLM이 허용된 범위 내에서 파라미터를 조정할 수 있게 한다."""

    def __init__(self, config_dir: Path | None = None):
        self.config_dir = config_dir or CONFIG_DIR
        self._settings: dict = {}
        self._safety_rules: dict = {}
        self._trading_params: dict = {}
        self._load_all()

    def _load_yaml(self, filename: str) -> dict:
        path = self.config_dir / filename
        if not path.exists():
            logger.warning(f"Config file not found: {path}")
            return {}
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    def _save_yaml(self, filename: str, data: dict) -> None:
        path = self.config_dir / filename
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(data, f, allow_unicode=True, default_flow_style=False, sort_keys=False)

    def _load_all(self) -> None:
        self._settings = self._load_yaml("settings.yaml")
        self._safety_rules = self._load_yaml("safety-rules.yaml")
        self._trading_params = self._load_yaml("trading-params.yaml")
        logger.info("All config files loaded")

    def reload(self) -> None:
        self._load_all()

    @property
    def settings(self) -> dict:
        return self._settings

    @property
    def safety_rules(self) -> dict:
        return self._safety_rules

    @property
    def trading_params(self) -> dict:
        return self._trading_params

    def get(self, dotted_key: str, default: Any = None) -> Any:
        """점 표기법으로 설정값 접근. 예: 'system.mode', 'llm.model'"""
        parts = dotted_key.split(".")
        # settings에서 먼저 검색, 없으면 trading_params
        for source in (self._settings, self._trading_params, self._safety_rules):
            value = source
            try:
                for part in parts:
                    value = value[part]
                return value
            except (KeyError, TypeError):
                continue
        return default

    def get_broker_base_url(self) -> str:
        account_type = self.get("broker.account_type", "virtual")
        if account_type == "virtual":
            return self.get("broker.base_url_virtual")
        return self.get("broker.base_url_real")

    def get_broker_ws_url(self) -> str:
        account_type = self.get("broker.account_type", "virtual")
        if account_type == "virtual":
            return self.get("broker.ws_url_virtual")
        return self.get("broker.ws_url_real")

    def validate_adjustment(self, param: str, value: Any) -> tuple[bool, str]:
        """LLM이 제안한 파라미터 조정이 safety-rules 범위 내인지 검증."""
        limits = self._safety_rules.get("adjustable_limits", {})

        # 점 표기법 → 언더스코어 (safety-rules에서의 키 형식)
        limit_key = param.replace(".", "_")

        if limit_key in limits:
            limit = limits[limit_key]
            min_val = limit.get("min")
            max_val = limit.get("max")
            if min_val is not None and value < min_val:
                return False, f"{param}: {value} < 최소값 {min_val}"
            if max_val is not None and value > max_val:
                return False, f"{param}: {value} > 최대값 {max_val}"

        return True, "OK"

    def apply_adjustments(self, adjustments: list[dict]) -> list[dict]:
        """LLM이 제안한 config 조정 목록을 검증 후 적용.

        Returns:
            적용 결과 리스트 [{"param": ..., "value": ..., "applied": bool, "reason": str}]
        """
        results = []
        changed = False

        for adj in adjustments:
            param = adj.get("param", "")
            value = adj.get("value")
            reason = adj.get("reason", "")

            valid, msg = self.validate_adjustment(param, value)
            if not valid:
                results.append({
                    "param": param,
                    "value": value,
                    "applied": False,
                    "reason": f"Safety rule violation: {msg}",
                })
                logger.warning(f"Config adjustment blocked: {param}={value} — {msg}")
                continue

            # trading-params.yaml에 적용
            parts = param.split(".")
            target = self._trading_params
            try:
                for part in parts[:-1]:
                    target = target[part]
                old_value = target.get(parts[-1])
                target[parts[-1]] = value
                changed = True
                results.append({
                    "param": param,
                    "value": value,
                    "old_value": old_value,
                    "applied": True,
                    "reason": reason,
                })
                logger.info(f"Config adjusted: {param}: {old_value} → {value} ({reason})")
            except (KeyError, TypeError) as e:
                results.append({
                    "param": param,
                    "value": value,
                    "applied": False,
                    "reason": f"Parameter not found: {e}",
                })

        if changed:
            self._save_yaml("trading-params.yaml", self._trading_params)
            logger.info("trading-params.yaml saved")

        return results

    def get_mode(self) -> str:
        return self.get("system.mode", "simulation")

    def set_mode(self, mode: str) -> None:
        if mode not in ("simulation", "live"):
            raise ValueError(f"Invalid mode: {mode}")
        self._settings["system"]["mode"] = mode
        self._save_yaml("settings.yaml", self._settings)
        logger.info(f"System mode changed to: {mode}")
