from __future__ import annotations

import json

from click.testing import CliRunner

from rag.cli import main


class TestCliMcpConfigPrint:
    def test_mcp_config_print_valid_json(self) -> None:
        """Run rag mcp-config --print and get valid JSON."""
        runner = CliRunner()
        result = runner.invoke(main, ["mcp-config", "--print"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert isinstance(data, dict)


class TestCliDoctor:
    def test_doctor_without_config(self) -> None:
        """Run rag doctor without config file — should report FAIL for config."""
        runner = CliRunner(env={"RAG_CONFIG_PATH": "/nonexistent/config.toml"})
        result = runner.invoke(main, ["doctor"])
        assert result.exit_code == 0
        assert "FAIL" in result.output or "Config" in result.output
