def test_agent_specs_import():
    from bloodcancer.config import AppConfig, load_yaml
    from bloodcancer.agents.agents import list_agent_specs
    cfg = AppConfig(load_yaml("configs/app.yaml"))
    specs = list_agent_specs(cfg)
    assert isinstance(specs, list) and len(specs) >= 5
    assert "name" in specs[0] and "proposition" in specs[0]
