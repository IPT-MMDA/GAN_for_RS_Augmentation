label_map = {
    0: 9,
    1: 0,
    2: 1,
    3: 2,
    4: 9,
    5: 3,
    6: 4,
    7: 5,
    8: 6,
    9: 9,
    10: 7,
    11: 8,
    12: 10,
    13: 11,
    14: 12,
    15: 13,
    16: 14,
    17: 9,
    18: 15,
    19: 9}
ind2crop_t = {
    2: "Wheat",
    3: "Rapeseed",
    4: "Buckwheat",
    5: "Maize",
    6: "Sugar beet",
    7: "Sunflower",
    8: "Soybeans",
    9: "Other crops",
    15: "Barley",
    16: "Peas",
}
ind2crop = {}
for k in ind2crop_t.keys():
    ind2crop[label_map[k]] = ind2crop_t[k]

ind2label_t = {
    1: "Artificial",
    2: "Wheat",
    3: "Rapeseed",
    5: "Maize",
    6: "Sugar beet",
    7: "Sunflower",
    8: "Soybeans",
    9: "Other crops",
    10: "Forest",
    11: "Grassland",
    12: "Bare land",
    13: "Water",
    14: "Wetland",
    15: "Barley",
    16: "Peas",
    18: "Gardens, parks",
}
ind2label = {}
for k in ind2label_t.keys():
    ind2label[label_map[k]] = ind2label_t[k]
