parameters = {
    "route.global.effort_level": ["minimum", "low", "medium", "high", "ultra"],
    "route.global.crosstalk_driven": ["true", "false"],
    "route.global.timing_driven": ["true", "false"],
    "route.global.timing_driven_effort_level": ["low", "high"],
    "route.track.crosstalk_driven": ["true", "false"],
    "route.track.timing_driven": ["true", "false"],
    "route.detail.timing_driven": ["true", "false"],
    "route.common.rc_driven_setup_effort_level": ["off", "low", "medium", "high"]
}

stage_parameters = {
    "place_opt": [
        "ccd.timing_effort",
        "ccd.max_prepone",
        "ccd.max_postpone",
        "place_opt.initial_place.buffering_aware",
        "place_opt.initial_drc.global_route_based",
    ],
    "clock_opt": [
        "route.global.effort_level",
        "route.global.timing_driven",
        "route.global.timing_driven_effort_level",
        "route.global.crosstalk_driven",
        "route.track.timing_driven",
        "route.track.crosstalk_driven",
        "route.detail.timing_driven",
        "route.common.rc_driven_setup_effort_level"
    ],
    "route_auto": [
        "route.track.timing_driven",
        "route.track.crosstalk_driven",
        "route.detail.timing_driven",
    ],
    "route_opt": [
        "route_opt.flow.enable_power",
        "route_opt.flow.enable_irdrivenopt"
    ]
}

conti_values = {
    "ccd.max_prepone": [0, 2],
    "ccd.max_postpone": [0, 2]
}

discrete_values = {
    "ccd.timing_effort": ["low", "medium", "high"],
    "place_opt.initial_place.buffering_aware": ["true", "false"],
    "place_opt.initial_drc.global_route_based": [0, 1],
    "route.global.effort_level": ["minimum", "low", "medium", "high", "ultra"],
    "route.global.crosstalk_driven": ["true", "false"],
    "route.global.timing_driven": ["true", "false"],
    "route.global.timing_driven_effort_level": ["low", "high"],
    "route.track.crosstalk_driven": ["true", "false"],
    "route.track.timing_driven": ["true", "false"],
    "route.detail.timing_driven": ["true", "false"],
    "route_opt.flow.enable_power": ["true", "false"],
    "route_opt.flow.enable_irdrivenopt": ["true", "false"],
    "route.common.rc_driven_setup_effort_level": ["off", "low", "medium", "high"]
}

convert_stage_name = {
        "initial_opto": "initial_opto",
        "place_opt": "final_opto", 
        "clock_opt": "clock_final_opto", 
        "route_auto": "route_auto", 
        "route_opt": "route_opt3"
}

default_freqs = {
    "aes": 4.5,
    "ldpc": 1.7,
    "vga": 1.85
}