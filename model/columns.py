# Patrick Berger, 2024

base_columns = ["beta", "u", "e_1", "e_2", "e_3", "e_4", "e_5", "v_1", "v_2", "v_3", "v_4", "v_5", "hf", "occupation"]

tau_columns = [
    "tau_1",
    "tau_2",
    "tau_3",
    "tau_4",
    "tau_5",
    "tau_6",
    "tau_7",
    "tau_8",
    "tau_9",
    "tau_10",
    "tau_11",
    "tau_12",
    "tau_13",
    "tau_14",
    "tau_15",
    "tau_16",
    "tau_17",
    "tau_18",
    "tau_19",
    "tau_20",
    "tau_21",
    "tau_22",
    "tau_23",
    "tau_24",
    "tau_25",
    "tau_26",
    "tau_27",
    "tau_28",
    "tau_29",
    "tau_30",
    "tau_31",
    "tau_32",
    "tau_33",
    "tau_34",
    "tau_35",
    "tau_36",
    "tau_37",
    "tau_38",
    "tau_39",
    "tau_40",
    "tau_41",
    "tau_42",
    "tau_43",
    "tau_44",
    "tau_45",
    "tau_46",
    "tau_47",
    "tau_48",
    "tau_49",
    "tau_50",
    "tau_51",
    "tau_52",
    "tau_53",
    "tau_54",
    "tau_55",
    "tau_56",
    "tau_57",
    "tau_58",
    "tau_59",
    "tau_60",
    "tau_61",
    "tau_62",
    "tau_63",
    "tau_64",
    "tau_65",
    "tau_66",
    "tau_67",
    "tau_68",
    "tau_69",
    "tau_70",
    "tau_71",
    "tau_72",
    "tau_73",
    "tau_74",
    "tau_75",
    "tau_76",
    "tau_77",
    "tau_78",
    "tau_79",
    "tau_80",
    "tau_81",
    "tau_82",
    "tau_83",
    "tau_84",
    "tau_85",
    "tau_86",
    "tau_87",
    "tau_88",
    "tau_89",
    "tau_90",
    "tau_91",
    "tau_92",
    "tau_93",
    "tau_94",
    "tau_95",
    "tau_96",
    "tau_97",
    "tau_98",
    "tau_99",
    "tau_100",
    "tau_101",
    "tau_102",
    "tau_103",
    "tau_104",
    "tau_105",
    "tau_106",
    "tau_107",
    "tau_108",
    "tau_109",
    "tau_110",
    "tau_111",
    "tau_112",
    "tau_113",
    "tau_114",
    "tau_115",
    "tau_116",
    "tau_117",
    "tau_118",
    "tau_119",
    "tau_120",
    "tau_121",
    "tau_122",
]

hyb_tau_columns = [
    "hyb_tau_1",
    "hyb_tau_2",
    "hyb_tau_3",
    "hyb_tau_4",
    "hyb_tau_5",
    "hyb_tau_6",
    "hyb_tau_7",
    "hyb_tau_8",
    "hyb_tau_9",
    "hyb_tau_10",
    "hyb_tau_11",
    "hyb_tau_12",
    "hyb_tau_13",
    "hyb_tau_14",
    "hyb_tau_15",
    "hyb_tau_16",
    "hyb_tau_17",
    "hyb_tau_18",
    "hyb_tau_19",
    "hyb_tau_20",
    "hyb_tau_21",
    "hyb_tau_22",
    "hyb_tau_23",
    "hyb_tau_24",
    "hyb_tau_25",
    "hyb_tau_26",
    "hyb_tau_27",
    "hyb_tau_28",
    "hyb_tau_29",
    "hyb_tau_30",
    "hyb_tau_31",
    "hyb_tau_32",
    "hyb_tau_33",
    "hyb_tau_34",
    "hyb_tau_35",
    "hyb_tau_36",
    "hyb_tau_37",
    "hyb_tau_38",
    "hyb_tau_39",
    "hyb_tau_40",
    "hyb_tau_41",
    "hyb_tau_42",
    "hyb_tau_43",
    "hyb_tau_44",
    "hyb_tau_45",
    "hyb_tau_46",
    "hyb_tau_47",
    "hyb_tau_48",
    "hyb_tau_49",
    "hyb_tau_50",
    "hyb_tau_51",
    "hyb_tau_52",
    "hyb_tau_53",
    "hyb_tau_54",
    "hyb_tau_55",
    "hyb_tau_56",
    "hyb_tau_57",
    "hyb_tau_58",
    "hyb_tau_59",
    "hyb_tau_60",
    "hyb_tau_61",
    "hyb_tau_62",
    "hyb_tau_63",
    "hyb_tau_64",
    "hyb_tau_65",
    "hyb_tau_66",
    "hyb_tau_67",
    "hyb_tau_68",
    "hyb_tau_69",
    "hyb_tau_70",
    "hyb_tau_71",
    "hyb_tau_72",
    "hyb_tau_73",
    "hyb_tau_74",
    "hyb_tau_75",
    "hyb_tau_76",
    "hyb_tau_77",
    "hyb_tau_78",
    "hyb_tau_79",
    "hyb_tau_80",
    "hyb_tau_81",
    "hyb_tau_82",
    "hyb_tau_83",
    "hyb_tau_84",
    "hyb_tau_85",
    "hyb_tau_86",
    "hyb_tau_87",
    "hyb_tau_88",
    "hyb_tau_89",
    "hyb_tau_90",
    "hyb_tau_91",
    "hyb_tau_92",
    "hyb_tau_93",
    "hyb_tau_94",
    "hyb_tau_95",
    "hyb_tau_96",
    "hyb_tau_97",
    "hyb_tau_98",
    "hyb_tau_99",
    "hyb_tau_100",
    "hyb_tau_101",
    "hyb_tau_102",
    "hyb_tau_103",
    "hyb_tau_104",
    "hyb_tau_105",
    "hyb_tau_106",
    "hyb_tau_107",
    "hyb_tau_108",
    "hyb_tau_109",
    "hyb_tau_110",
    "hyb_tau_111",
    "hyb_tau_112",
    "hyb_tau_113",
    "hyb_tau_114",
    "hyb_tau_115",
    "hyb_tau_116",
    "hyb_tau_117",
    "hyb_tau_118",
    "hyb_tau_119",
    "hyb_tau_120",
    "hyb_tau_121",
    "hyb_tau_122",
]

sigma_tau_columns = [
    "sigma_tau_1",
    "sigma_tau_2",
    "sigma_tau_3",
    "sigma_tau_4",
    "sigma_tau_5",
    "sigma_tau_6",
    "sigma_tau_7",
    "sigma_tau_8",
    "sigma_tau_9",
    "sigma_tau_10",
    "sigma_tau_11",
    "sigma_tau_12",
    "sigma_tau_13",
    "sigma_tau_14",
    "sigma_tau_15",
    "sigma_tau_16",
    "sigma_tau_17",
    "sigma_tau_18",
    "sigma_tau_19",
    "sigma_tau_20",
    "sigma_tau_21",
    "sigma_tau_22",
    "sigma_tau_23",
    "sigma_tau_24",
    "sigma_tau_25",
    "sigma_tau_26",
    "sigma_tau_27",
    "sigma_tau_28",
    "sigma_tau_29",
    "sigma_tau_30",
    "sigma_tau_31",
    "sigma_tau_32",
    "sigma_tau_33",
    "sigma_tau_34",
    "sigma_tau_35",
    "sigma_tau_36",
    "sigma_tau_37",
    "sigma_tau_38",
    "sigma_tau_39",
    "sigma_tau_40",
    "sigma_tau_41",
    "sigma_tau_42",
    "sigma_tau_43",
    "sigma_tau_44",
    "sigma_tau_45",
    "sigma_tau_46",
    "sigma_tau_47",
    "sigma_tau_48",
    "sigma_tau_49",
    "sigma_tau_50",
    "sigma_tau_51",
    "sigma_tau_52",
    "sigma_tau_53",
    "sigma_tau_54",
    "sigma_tau_55",
    "sigma_tau_56",
    "sigma_tau_57",
    "sigma_tau_58",
    "sigma_tau_59",
    "sigma_tau_60",
    "sigma_tau_61",
    "sigma_tau_62",
    "sigma_tau_63",
    "sigma_tau_64",
    "sigma_tau_65",
    "sigma_tau_66",
    "sigma_tau_67",
    "sigma_tau_68",
    "sigma_tau_69",
    "sigma_tau_70",
    "sigma_tau_71",
    "sigma_tau_72",
    "sigma_tau_73",
    "sigma_tau_74",
    "sigma_tau_75",
    "sigma_tau_76",
    "sigma_tau_77",
    "sigma_tau_78",
    "sigma_tau_79",
    "sigma_tau_80",
    "sigma_tau_81",
    "sigma_tau_82",
    "sigma_tau_83",
    "sigma_tau_84",
    "sigma_tau_85",
    "sigma_tau_86",
    "sigma_tau_87",
    "sigma_tau_88",
    "sigma_tau_89",
    "sigma_tau_90",
    "sigma_tau_91",
    "sigma_tau_92",
    "sigma_tau_93",
    "sigma_tau_94",
    "sigma_tau_95",
    "sigma_tau_96",
    "sigma_tau_97",
    "sigma_tau_98",
    "sigma_tau_99",
    "sigma_tau_100",
    "sigma_tau_101",
    "sigma_tau_102",
    "sigma_tau_103",
    "sigma_tau_104",
    "sigma_tau_105",
    "sigma_tau_106",
    "sigma_tau_107",
    "sigma_tau_108",
    "sigma_tau_109",
    "sigma_tau_110",
    "sigma_tau_111",
    "sigma_tau_112",
    "sigma_tau_113",
    "sigma_tau_114",
    "sigma_tau_115",
    "sigma_tau_116",
    "sigma_tau_117",
    "sigma_tau_118",
    "sigma_tau_119",
    "sigma_tau_120",
    "sigma_tau_121",
    "sigma_tau_122",
]

so_columns = [
    "so_1",
    "so_2",
    "so_3",
    "so_4",
    "so_5",
    "so_6",
    "so_7",
    "so_8",
    "so_9",
    "so_10",
    "so_11",
    "so_12",
    "so_13",
    "so_14",
    "so_15",
    "so_16",
    "so_17",
    "so_18",
    "so_19",
    "so_20",
    "so_21",
    "so_22",
    "so_23",
    "so_24",
    "so_25",
    "so_26",
    "so_27",
    "so_28",
    "so_29",
    "so_30",
    "so_31",
    "so_32",
    "so_33",
    "so_34",
    "so_35",
    "so_36",
    "so_37",
    "so_38",
    "so_39",
    "so_40",
    "so_41",
    "so_42",
    "so_43",
    "so_44",
    "so_45",
    "so_46",
    "so_47",
    "so_48",
    "so_49",
    "so_50",
    "so_51",
    "so_52",
    "so_53",
    "so_54",
    "so_55",
    "so_56",
    "so_57",
    "so_58",
    "so_59",
    "so_60",
    "so_61",
    "so_62",
    "so_63",
    "so_64",
    "so_65",
    "so_66",
    "so_67",
    "so_68",
    "so_69",
    "so_70",
    "so_71",
    "so_72",
    "so_73",
    "so_74",
    "so_75",
    "so_76",
    "so_77",
    "so_78",
    "so_79",
    "so_80",
    "so_81",
    "so_82",
    "so_83",
    "so_84",
    "so_85",
    "so_86",
    "so_87",
    "so_88",
    "so_89",
    "so_90",
    "so_91",
    "so_92",
    "so_93",
    "so_94",
    "so_95",
    "so_96",
    "so_97",
    "so_98",
    "so_99",
    "so_100",
    "so_101",
    "so_102",
    "so_103",
    "so_104",
    "so_105",
    "so_106",
    "so_107",
    "so_108",
    "so_109",
    "so_110",
    "so_111",
    "so_112",
    "so_113",
    "so_114",
    "so_115",
    "so_116",
    "so_117",
    "so_118",
    "so_119",
    "so_120",
    "so_121",
    "so_122",
]

encoder_input = base_columns + so_columns

labels = sigma_tau_columns
