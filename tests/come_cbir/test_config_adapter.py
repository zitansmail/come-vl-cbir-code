from come_cbir.config import RunConfig


def test_adapter_block_parses_from_yaml(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
dataset:
  type: folder
  root: /data/corel1000

model:
  checkpoint: /models/come-vl
  device: cpu
  dtype: float32

experiments:
  - name: siglip_mean
    descriptor_mode: siglip
    pooling: mean
  - name: siglip_adapted
    descriptor_mode: siglip
    pooling: mean
    adapter:
      enabled: true
      type: arp
      output_dim: 256
"""
    )
    config = RunConfig.from_yaml(str(config_path))

    assert config.experiments[0].adapter is None
    adapter_entry = config.experiments[1].adapter
    assert adapter_entry is not None
    assert adapter_entry.enabled is True
    assert adapter_entry.type == "arp"
    assert adapter_entry.output_dim == 256
    assert adapter_entry.hidden_dim == 256  # default, not specified in YAML
