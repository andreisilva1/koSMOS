from fastapi import HTTPException


def check_dict_values(dict_types: dict, dict_values: dict):
    error_list = []
    for key, value in dict_types.items():

        if key not in dict_values or dict_values[key] in ["", None]:
            error_list.append(f"{key} needs to be provided.")
            continue

        val = dict_values[key]

        if value["col_type"] == "int":
            try:
                dict_values[key] = int(val)
            except ValueError:
                error_list.append(f"{key} needs to be an integer.")

        if value["col_type"] == "float":
            try:
                dict_values[key] = float(val)
            except ValueError:
                error_list.append(f"{key} needs to be a number.")

        if value["col_type"] in ["enum", "ordinal"]:
            if str(val) not in value["values"]:
                error_list.append(
                    f"{key} needs to be one of the options: {value['values']}."
                )

        if value["col_type"] == "range":

            try:
                num = float(val)
            except ValueError:
                error_list.append(f"{key} needs to be a number.")
                continue

            min_val = value["values"][0]
            max_val = value["values"][1]
            must_be_int = value["values"][2]

            if must_be_int == 1:
                if not float(val).is_integer():
                    error_list.append(f"{key} needs to be an integer.")

            if min_val is not None:
                if num < float(min_val):
                    error_list.append(f"{key} needs to be more than {min_val}.")

            if max_val is not None:
                if num > float(max_val):
                    error_list.append(f"{key} needs to be less than {max_val}.")

    if error_list:
        raise HTTPException(status_code=400, detail={"error_list": error_list})
