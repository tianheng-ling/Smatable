def get_DataByPerson(
    data_splitting_method: str,
    target_subject: str,
) -> tuple:
    behaviors_list = ["UP", "DOWN", "RIGHT", "LEFT"]
    if data_splitting_method == "PS":  # per subject
        if target_subject == "A":
            train_subjects_list = ["ID4", "ID5", "ID6", "ID7", "ID8", "ID9"]
            test_subjects_list = ["ID1", "ID2", "ID3"]
        elif target_subject == "B":
            train_subjects_list = ["ID24", "ID25", "ID26", "ID27", "ID28", "ID29"]
            test_subjects_list = ["ID21", "ID22", "ID23"]
        elif target_subject == "C":
            train_subjects_list = ["ID34", "ID35", "ID36", "ID37", "ID38", "ID39"]
            test_subjects_list = ["ID31", "ID32", "ID33"]
        else:
            raise ValueError("Invalid target_subject subject")
    elif data_splitting_method == "LOSO":  # leave one subject out
        if target_subject == "A":
            train_subjects_list = ["ID21", "ID22", "ID23", "ID31", "ID32", "ID33"]
            test_subjects_list = ["ID1", "ID2", "ID3"]
        elif target_subject == "B":
            train_subjects_list = ["ID1", "ID2", "ID3", "ID31", "ID32", "ID33"]
            test_subjects_list = ["ID21", "ID22", "ID23"]
        elif target_subject == "C":
            train_subjects_list = ["ID1", "ID2", "ID3", "ID21", "ID22", "ID23"]
            test_subjects_list = ["ID31", "ID32", "ID33"]
        else:
            raise ValueError("Invalid target_subject subject")
    elif data_splitting_method == "AOS":  # add one session from the targt subject
        if target_subject == "A":
            train_subjects_list = [
                "ID4",
                "ID21",
                "ID22",
                "ID23",
                "ID31",
                "ID32",
                "ID33",
            ]
            test_subjects_list = ["ID1", "ID2", "ID3"]
        elif target_subject == "B":
            train_subjects_list = ["ID24", "ID1", "ID2", "ID3", "ID31", "ID32", "ID33"]
            test_subjects_list = ["ID21", "ID22", "ID23"]
        elif target_subject == "C":
            train_subjects_list = ["ID34", "ID1", "ID2", "ID3", "ID21", "ID22", "ID23"]
            test_subjects_list = ["ID31", "ID32", "ID33"]
        else:
            raise ValueError("Invalid target_subject subject")
    else:
        raise ValueError("Invalid data splitting method")

    return behaviors_list, train_subjects_list, test_subjects_list


def get_DataByTable(
    data_splitting_method: str,
    target_subject: str,
) -> tuple:
    behaviors_list = ["UP", "DOWN", "RIGHT", "LEFT"]
    if data_splitting_method == "PS":  # per subject
        if target_subject == "A":
            train_subjects_list = ["ID4", "ID5", "ID6", "ID7", "ID8", "ID9"]
            test_subjects_list = ["ID1", "ID2", "ID3"]
        elif target_subject == "B":
            train_subjects_list = ["ID64", "ID65", "ID66", "ID67", "ID68", "ID69"]
            test_subjects_list = ["ID61", "ID62", "ID63"]
        elif target_subject == "C":
            train_subjects_list = ["ID94", "ID95", "ID96", "ID97", "ID98", "ID99"]
            test_subjects_list = ["ID91", "ID92", "ID93"]
        else:
            raise ValueError("Invalid target_subject subject")
    elif data_splitting_method == "LOSO":  # leave one subject out
        if target_subject == "A":
            train_subjects_list = ["ID61", "ID62", "ID63", "ID91", "ID92", "ID93"]
            test_subjects_list = ["ID1", "ID2", "ID3"]
        elif target_subject == "B":
            train_subjects_list = ["ID1", "ID2", "ID3", "ID91", "ID92", "ID93"]
            test_subjects_list = [
                "ID61",
                "ID62",
                "ID63",
            ]
        elif target_subject == "C":
            train_subjects_list = ["ID1", "ID2", "ID3", "ID61", "ID62", "ID63"]
            test_subjects_list = ["ID91", "ID92", "ID93"]
        else:
            raise ValueError("Invalid target_subject subject")
    elif data_splitting_method == "AOS":  # add one session from the targt subject
        if target_subject == "A":
            train_subjects_list = [
                "ID4",
                "ID61",
                "ID62",
                "ID63",
                "ID91",
                "ID92",
                "ID93",
            ]
            test_subjects_list = ["ID1", "ID2", "ID3"]
        elif target_subject == "B":
            train_subjects_list = [
                "ID64",
                "ID1",
                "ID2",
                "ID3",
                "ID91",
                "ID92",
                "ID93",
            ]
            test_subjects_list = [
                "ID61",
                "ID62",
                "ID63",
            ]
        elif target_subject == "C":
            train_subjects_list = [
                "ID94",
                "ID1",
                "ID2",
                "ID3",
                "ID61",
                "ID62",
                "ID63",
            ]
            test_subjects_list = ["ID91", "ID92", "ID93"]
        else:
            raise ValueError("Invalid target_subject subject")
    else:
        raise ValueError("Invalid data splitting method")

    return behaviors_list, train_subjects_list, test_subjects_list


def get_data(
    data_flag: str,
    data_splitting_method: str,
    target_subject: str,
) -> tuple:
    data_flag_options = {
        "DatabyPerson": get_DataByPerson,
        "DatabyTable": get_DataByTable,
    }
    if data_flag not in data_flag_options:
        raise ValueError(f"Invalid data flag: {data_flag}")

    return data_flag_options[data_flag](data_splitting_method, target_subject)
