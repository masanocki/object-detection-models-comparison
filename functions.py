import json


def set_model_details(
    selected_model, model_name_var, model_version_var, model_description_var
):
    with open("models_details.json", "r") as file:
        models_details = json.load(file)

    if selected_model in models_details:
        model_details = models_details[selected_model]
        model_name_var.set(model_details["model_name"])
        model_version_var.set(model_details["model_version"])
        model_description_var.set(model_details["model_description"])
    else:
        model_name_var.set("None")
        model_version_var.set("None")
        model_description_var.set("None")
