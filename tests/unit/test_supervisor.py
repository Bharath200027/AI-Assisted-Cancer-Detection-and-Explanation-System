def test_supervisor_plan():
    from bloodcancer.config import AppConfig, load_yaml
    from bloodcancer.agents.supervisor import SupervisorAgent
    cfg = AppConfig(load_yaml("configs/app.yaml"))
    sup = SupervisorAgent(cfg)
    plan = sup.propose_plan({"image_path": "x.png"})
    assert "triage" in plan and "inference" in plan and "report" in plan and "safety_gate" in plan
