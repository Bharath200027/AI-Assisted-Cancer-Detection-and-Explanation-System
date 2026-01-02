def test_routing_parses_models_yaml():
    from bloodcancer.config import AppConfig, load_yaml
    from bloodcancer.agents.tools import RoutingTool
    cfg = AppConfig(load_yaml("configs/app.yaml"))
    rt = RoutingTool(cfg)
    st = rt.run({"image_path": "x.png"})
    assert "policy" in st and "stage1_candidates" in st

def test_subtype_noop_when_not_leukemia():
    from bloodcancer.config import AppConfig, load_yaml
    from bloodcancer.agents.tools import SubtypeInferenceTool
    cfg = AppConfig(load_yaml("configs/app.yaml"))
    tool = SubtypeInferenceTool(cfg)
    out = tool.run({"mode":"two_stage", "predicted_label":"normal", "stage2_candidates":[{"model_name":"x","checkpoint":"y","class_names":["all","aml","cll","cml"]}], "stage2_ensemble":False})
    assert out == {}
