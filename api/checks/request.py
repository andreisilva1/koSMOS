from fastapi import HTTPException


def check_dict_values(dict_types: dict, dict_values: dict):
    error_list = []
    for key, value in dict_types.items():
        if not dict_values[key]:
            error_list.append(f"{key} needs to be provided.")
            continue

        if value["col_type"] == "int":
            try:
                dict_values[key] = int(dict_values[key])
            except:
                error_list.append(f"{key} needs to be integer.")

        if (
            value["col_type"] == "float"
            and not isinstance(dict_values[key], int)
            and not isinstance(dict_values[key], float)
        ):
            try:
                dict_values[key] = float(dict_values[key])
            except:
                error_list.append(f"{key} needs to be a number.")

        if (
            value["col_type"] in ["enum", "ordinal"]
            and dict_values[key] not in value["values"]
        ):
            error_list.append(
                f"{key} needs to be one of the options: {value['values']}."
            )

        if value["col_type"] == "range":
            if value["values"][2] == 1 and type(dict_values[key] == float):
                error_list.append(f"{key} needs to be a integer.")

            elif value["values"][2] == 0:
                if value["values"][1]:
                    if dict_values[key] > value["values"][1]:
                        error_list.append(
                            f"{key} needs to be less than {value["values"][1]}."
                        )
                if value["values"][2]:
                    if dict_values[key] < value["values"][0]:
                        error_list.append(
                            f"{key} needs to be more than {value["values"][1]}."
                        )

    if error_list:
        raise HTTPException(status_code=400, detail={"error_list": error_list})
