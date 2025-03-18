
from core.elastic import client
from models.index_template import Template


def create_indices(template: Template):
    # template_dict = json.loads(template)
    # settings = template_dict.get("settings", {})
    # properties = template_dict.get("mappings", {}).get("properties", {})
    #
    # print("Settings:", settings)
    # print("Properties:", properties)
    print(template)

