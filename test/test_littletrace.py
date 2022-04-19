import littletrace
from pathlib import Path
import json

PATH_CONFIG_JSON = Path(__file__).parent.parent  / "config.template.json"

def test_config():
    config = littletrace.TraceConfig(**json.loads(PATH_CONFIG_JSON.read_text()))

