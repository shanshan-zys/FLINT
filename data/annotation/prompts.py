"""
数据标注 Prompt 模板

三层标注策略:
1. 场景描述 (scenario): 描述背景/可行走区域, 只需标注一次/场景
2. 人群动态 (crowd): 描述每个clip的人群行为, 每个clip一次
3. 质量检查 (quality): 用于验证标注质量

成本估算 (以1000个clip为例):
┌──────────────────────┬────────────────┬───────────────┬──────────────┐
│  方案                 │  每clip成本     │  1000clip总价  │  质量         │
├──────────────────────┼────────────────┼───────────────┼──────────────┤
│  GPT-4o (图像)        │  ~$0.02        │  ~$20         │  最优         │
│  GPT-4o-mini (图像)   │  ~$0.003       │  ~$3          │  良好         │
│  Gemini-1.5-Flash     │  ~$0.001       │  ~$1          │  良好         │
│  本地 Qwen2.5-VL-7B  │  免费(需GPU)    │  $0           │  可用         │
│  混合: 本地初筛+API精修 │ ~$0.005       │  ~$5          │  接近最优     │
└──────────────────────┴────────────────┴───────────────┴──────────────┘

推荐策略: Gemini-1.5-Flash 做主力标注 (极便宜), GPT-4o 做10%抽样质检
"""

# ====================================================================
# Prompt 1: 场景描述 (每个场景只需调用一次)
# ====================================================================
SCENARIO_DESCRIPTION_PROMPT = """You are an expert in pedestrian scene analysis. Given a background image of a public space, describe the scenario in ONE concise sentence.

Focus on:
- The type of location (university campus, hotel entrance, street, intersection, etc.)
- Walkable areas and their layout (sidewalks, plazas, pathways)
- Key landmarks or obstacles (buildings, fences, trees, parked vehicles)

Output format: A single sentence, no more than 50 words.

Example: "A university campus plaza with a wide diagonal walkway crossing from bottom-left to top-right, bordered by grass areas and a building entrance on the right side."
"""

# ====================================================================
# Prompt 2: 人群动态描述 (每个clip调用一次)
# ====================================================================
CROWD_DYNAMICS_PROMPT = """You are analyzing a {duration}-second pedestrian video clip ({num_frames} frames at {fps} FPS).

The scene contains {num_pedestrians} pedestrians. Their trajectories (in pixel coordinates) are provided below.

Pedestrian trajectories (format: ID: [(x1,y1), (x2,y2), ...]):
{trajectory_text}

Write a detailed paragraph (100-150 words) describing the crowd dynamics. Cover ALL of the following aspects:

1. **Overall density and population**: How many people, how crowded
2. **Dominant flow directions**: Which directions do most people move (e.g., left-to-right, top-to-bottom)
3. **Sources and destinations**: Where do people enter/exit the scene
4. **Collective dynamics**: Any group behaviors, lane formation, clustering
5. **Key events**: Any notable interactions (people meeting, splitting, stopping, avoiding each other)
6. **Temporal changes**: How does the crowd pattern change over the clip duration
7. **Speed patterns**: Are people walking fast, slow, or mixed

Be specific about directions and positions. Use terms like "upper-left", "center", "bottom-right" for spatial references.
"""

# ====================================================================
# Prompt 2b: 纯视觉描述 (有视频帧时使用, 更丰富)
# ====================================================================
CROWD_DYNAMICS_VISUAL_PROMPT = """Analyze these {num_frames} sequential frames from a pedestrian surveillance video.

Scene context: {scenario_description}
Number of tracked pedestrians: {num_pedestrians}

Write a detailed paragraph (100-150 words) describing the crowd dynamics you observe. Cover:

1. Overall crowd density and how it changes
2. Dominant movement directions and flow patterns
3. Where pedestrians enter and exit the scene
4. Any group behaviors, interactions, or notable events
5. Speed patterns (fast walkers, slow walkers, stationary people)
6. Any unusual or interesting behaviors

Be specific about spatial locations using terms like "upper-left", "center", "bottom-right".
"""

# ====================================================================
# Prompt 3: 低成本纯轨迹描述 (不需要图像, 最便宜)
# ====================================================================
CROWD_DYNAMICS_TEXT_ONLY_PROMPT = """You are given pedestrian trajectory data from a crowd simulation dataset.

Scene: {scenario_description}
Image resolution: {width}x{height} pixels
Duration: {duration} seconds ({num_frames} frames)

Pedestrian trajectories (ID: frame_start-frame_end, positions):
{trajectory_text}

Based on the trajectory coordinates, write a detailed paragraph (100-150 words) describing the crowd dynamics. Analyze:

1. How many pedestrians and overall density
2. Main movement directions (compute from trajectory start→end vectors)
3. Entry/exit regions of the scene
4. Any interactions: near-misses (distance < 20px), groups moving together (similar trajectories), people stopping (small displacement)
5. Speed distribution (compute from frame-to-frame displacement)
6. Temporal evolution of the scene

This is the CHEAPEST annotation approach — no image tokens needed, only text.
"""

# ====================================================================
# Prompt 4: 多样性增强描述 (为同一场景生成多样化text)
# ====================================================================
DIVERSE_DESCRIPTION_PROMPT = """Given the following crowd scenario, generate {num_variants} DIFFERENT but plausible textual descriptions that would produce DIVERSE crowd behaviors in this scene.

Original scenario: {scenario_description}
Original crowd dynamics: {original_dynamics}

For each variant, modify ONE or MORE of these aspects:
- Crowd density (sparse ↔ dense)
- Movement speed (slow stroll ↔ fast walk ↔ running)
- Flow pattern (unidirectional ↔ bidirectional ↔ chaotic)
- Events (normal flow, evacuation, gathering, dispersal)
- Social behaviors (individuals ↔ groups ↔ queues)

Output format — return exactly {num_variants} descriptions, each 80-120 words, separated by "---":
"""

# ====================================================================
# Prompt 5: SFT Instruction (放在Alpaca格式的instruction字段)
# ====================================================================

SFT_INSTRUCTION_ABSOLUTE = """You are a crowd simulation engine. Given scene context and initial conditions, generate realistic pedestrian trajectories as coordinate tokens.

Output rules:
1. Output a SINGLE unbroken string of coordinate tokens, no spaces between tokens
2. Use <x_i><y_j> for each pedestrian's position at each timestep
3. Use <x_0><y_0> when a pedestrian has not appeared or has left the scene
4. Order: for each timestep t=1..T, output all N pedestrians' positions sequentially
5. Coordinate system: x increases rightward (1 to {x_bins}), y increases downward (1 to {y_bins})
6. Do NOT add any explanation, only output the token string"""

SFT_INSTRUCTION_RELATIVE = """You are a crowd simulation engine. Given scene context and initial conditions, generate realistic pedestrian trajectories using displacement tokens.

Output rules:
1. First timestep: output absolute positions <x_i><y_j> for all N pedestrians
2. Subsequent timesteps: output displacements <dx_i><dy_j> relative to previous positions
3. Use <x_0><y_0> or <dx_absent><dy_absent> for absent pedestrians
4. Output a SINGLE unbroken string, no spaces
5. Order: all N pedestrians per timestep, then next timestep
6. Do NOT add any explanation"""

SFT_INSTRUCTION_POLAR = """You are a crowd simulation engine. Given scene context and initial conditions, generate realistic pedestrian trajectories using polar coordinate tokens.

Output rules:
1. First timestep: output absolute positions <x_i><y_j> for all N pedestrians
2. Subsequent timesteps: output polar displacements <d_i><a_j> (distance and angle from previous position)
3. Use <a_still> when a pedestrian is stationary
4. Use <d_absent><a_absent> for absent pedestrians
5. Output a SINGLE unbroken string, no spaces
6. Do NOT add any explanation"""

# ====================================================================
# Prompt 6: SFT Input 模板 (放在Alpaca格式的input字段)
# ====================================================================
SFT_INPUT_TEMPLATE = """Walkable area (0=obstacle, 1=walkable, {grid_h}x{grid_w} grid):
{walkable_area}

Scenario: {scenario_description}

Crowd dynamics: {crowd_dynamics}

Number of pedestrians: {num_pedestrians}
Number of timesteps: {num_timesteps}

Initial states (ID: position, velocity):
{initial_states}"""

# ====================================================================
# 质量检查 Prompt
# ====================================================================
QUALITY_CHECK_PROMPT = """Rate the following crowd dynamics description on a scale of 1-5 for each criterion:

Description: {description}
Actual trajectory data: {trajectory_summary}

Criteria:
1. Accuracy: Does it match the actual trajectories? (1=completely wrong, 5=perfectly accurate)
2. Completeness: Does it cover density, direction, interactions, speed? (1=missing most, 5=all covered)
3. Specificity: Does it use concrete spatial references? (1=very vague, 5=very specific)

Output format (JSON):
{{"accuracy": X, "completeness": X, "specificity": X, "issues": "brief description of problems if any"}}
"""
