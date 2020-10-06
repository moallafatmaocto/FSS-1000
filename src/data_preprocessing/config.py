import os

TEST_CATEGORIES = {"bus", "hotel_slipper", "burj_al", "reflex_camera", "abe's_flyingfish", "oiltank_car", "doormat",
                   "fish_eagle", "barber_shaver", "motorbike", "feather_clothes", "wandering_albatross", "rice_cooker",
                   "delta_wing", "fish", "nintendo_switch", "bustard", "diver", "minicooper", "cathedrale_paris",
                   "big_ben", "combination_lock", "villa_savoye", "american_alligator", "gym_ball", "andean_condor",
                   "leggings", "pyramid_cube", "jet_aircraft", "meatloaf", "reel", "swan", "osprey", "crt_screen",
                   "microscope", "rubber_eraser", "arrow", "monkey", "mitten", "spiderman", "parthenon", "bat",
                   "chess_king", "sulphur_butterfly", "quail_egg", "oriole", "iron_man", "wooden_boat", "anise",
                   "steering_wheel", "groenendael", "dwarf_beans", "pteropus", "chalk_brush", "bloodhound", "moon",
                   "english_foxhound", "boxing_gloves", "peregine_falcon", "pyraminx", "cicada", "screw",
                   "shower_curtain", "tredmill", "bulb", "bell_pepper", "lemur_catta", "doughnut", "twin_tower",
                   "astronaut", "nintendo_3ds", "fennel_bulb", "indri", "captain_america_shield", "kunai", "broom",
                   "iphone", "earphone1", "flying_squirrel", "onion", "vinyl", "sydney_opera_house", "oyster",
                   "harmonica", "egg", "breast_pump", "guitar", "potato_chips", "tunnel", "cuckoo", "rubick_cube",
                   "plastic_bag", "phonograph", "net_surface_shoes", "goldfinch", "ipad", "mite_predator", "coffee_mug",
                   "golden_plover", "f1_racing", "lapwing", "nintendo_gba", "pizza", "rally_car", "drilling_platform",
                   "cd", "fly", "magpie_bird", "leaf_fan", "little_blue_heron", "carriage", "moist_proof_pad",
                   "flying_snakes", "dart_target", "warehouse_tray", "nintendo_wiiu", "chiffon_cake", "bath_ball",
                   "manatee", "cloud", "marimba", "eagle", "ruler", "soymilk_machine", "sled", "seagull",
                   "glider_flyingfish", "doublebus", "transport_helicopter", "window_screen", "truss_bridge", "wasp",
                   "snowman", "poached_egg", "strawberry", "spinach", "earphone2", "downy_pitch", "taj_mahal",
                   "rocking_chair", "cablestayed_bridge", "sealion", "banana_boat", "pheasant", "stone_lion",
                   "electronic_stove", "fox", "iguana", "rugby_ball", "hang_glider", "water_buffalo", "lotus",
                   "paper_plane", "missile", "flamingo", "american_chamelon", "kart", "chinese_knot",
                   "cabbage_butterfly", "key", "church", "tiltrotor", "helicopter", "french_fries", "water_heater",
                   "snow_leopard", "goblet", "fan", "snowplow", "leafhopper", "pspgo", "black_bear", "quail", "condor",
                   "chandelier", "hair_razor", "white_wolf", "toaster", "pidan", "pyramid", "chicken_leg",
                   "letter_opener", "apple_icon", "porcupine", "chicken", "stingray", "warplane", "windmill",
                   "bamboo_slip", "wig", "flying_geckos", "stonechat", "haddock", "australian_terrier", "hover_board",
                   "siamang", "canton_tower", "santa_sledge", "arch_bridge", "curlew", "sushi", "beet_root",
                   "accordion", "leaf_egg", "stealth_aircraft", "stork", "bucket", "hawk", "chess_queen", "ocarina",
                   "knife", "whippet", "cantilever_bridge", "may_bug", "wagtail", "leather_shoes", "wheelchair",
                   "shumai", "speedboat", "vacuum_cup", "chess_knight", "pumpkin_pie", "wooden_spoon",
                   "bamboo_dragonfly", "ganeva_chair", "soap", "clearwing_flyingfish", "pencil_sharpener1", "cricket",
                   "photocopier", "nintendo_sp", "samarra_mosque", "clam", "charge_battery", "flying_frog",
                   "ferrari911", "polo_shirt", "echidna", "coin", "tower_pisa"}

ROOT_PATH = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
DATA_PATH = os.path.join(ROOT_PATH, 'data')
TRAIN_DATA_PATH = os.path.join(DATA_PATH, 'train')
VAL_DATA_PATH = os.path.join(DATA_PATH, 'val')
TEST_DATA_PATH = os.path.join(DATA_PATH, 'test')

TRAIN_NUMBER = 520
SAMPLE_NUM_PER_CATEGORY = 5