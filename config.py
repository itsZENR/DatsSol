import os

TOKEN = os.getenv("DATSSOL_TOKEN", "")
BASE_URL = "https://games-test.datsteam.dev"

# Default game parameters (server values override where applicable)
DEFAULT_AR = 2
DEFAULT_SR = 3
DEFAULT_VR = 3
DEFAULT_MHP = 50
DEFAULT_CS = 5
DEFAULT_RS = 5
DEFAULT_SE = 5
DEFAULT_BE = 5
DEFAULT_DS = 10
BUILD_HP_TARGET = 50

# Upgrade sequence config.
# Format: one upgrade name per line, with a pattern string after ':'.
# Each character position = one level of that upgrade.
# Digit = global purchase step (1-based). '*' = not in sequence at that level.
# Supports steps 1-9. For 10-15, use letters a-f.
UPGRADE_SEQUENCE_SPEC = """\
settlement_limit:1**45*****
repair_power:*23
max_hp:**6**
beaver_damage_mitigation:*****7*
earthquake_mitigation:******8*
signal_range:*********9
"""

# Strategy weights (higher = higher priority in decision loop)
STRATEGY_WEIGHTS = {
    "upgrade":        100,
    "repair_hq":      90,
    "repair":         70,
    "expand":         60,
    "beaver_snipe":   50,
    "sabotage":       40,
    "beaver_attack":  30,
}

# Repair thresholds (fraction of MHP)
HQ_REPAIR_THRESHOLD = 0.7
REPAIR_THRESHOLD = 0.5

# Beaver snipe: attack on last turn when beaver HP is below this
BEAVER_SNIPE_HP = 20

# Min plantations before switching to aggressive strategies
MIN_PLANTATIONS_FOR_AGGRESSION = 5

# Loop timing
POLL_INTERVAL = 0.8  # seconds between get_arena calls
