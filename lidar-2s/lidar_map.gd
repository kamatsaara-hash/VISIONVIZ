extends Node2D


# ============================================================
# FILES
# ============================================================

const DATA_DIR: String = "res://data/"
const FRAME_DIGITS: int = 6
const POSES_PATH: String = "res://data/00/poses.txt"

@export var show_trajectory: bool = true
@export var trajectory_panel_position: Vector2 = Vector2(830.0, 25.0)
@export var trajectory_panel_size: Vector2 = Vector2(290.0, 175.0)
@export var trajectory_padding: float = 14.0

var trajectory_positions: Array[Vector3] = []

# Frames 000000 through 000500 in sequence 00.
const START_FRAME: int = 0
const END_FRAME: int = 500
const SEQUENCE_NAME: String = "00"

var FRAME_SEQUENCE: Array[Dictionary] = []

var current_sequence: String = SEQUENCE_NAME
var current_frame: int = START_FRAME
var current_frame_index: int = 0

func build_frame_sequence() -> void:
	FRAME_SEQUENCE.clear()

	for frame_number: int in range(START_FRAME, END_FRAME + 1):
		FRAME_SEQUENCE.append({
			"sequence": SEQUENCE_NAME,
			"frame": frame_number
		})

	print("Total frames configured: ", FRAME_SEQUENCE.size())


# Automatic playback interval (seconds per frame).
@export var frame_interval: float = 0.001
var playback_paused: bool = false
var playback_timer: float = 0.0

# Mouse-drag inspection controls (the existing 2.5D projection is retained).
@export var mouse_look_sensitivity: float = 0.002
@export var mouse_height_sensitivity: float = 0.005
var mouse_dragging: bool = false
var last_mouse_position: Vector2 = Vector2.ZERO


# ============================================================
# CAMERA
# ============================================================

@onready var camera: Camera2D = $Camera2D


# ============================================================
# SCREEN / EGO VIEW
# ============================================================

@export var map_center: Vector2 = Vector2(400.0, 300.0)

# Perspective focal length.
@export var focal_length: float = 520.0

# Height of the virtual camera above LiDAR origin.
@export var camera_height: float = 1.5

# Distance at which objects are considered visible.
@export var near_clip: float = 1.5

@export var far_clip: float = 80.0


# ============================================================
# PRECOMPUTED MESH
# ============================================================

# Mesh geometry is loaded from precomputed Poisson meshbin files.


# ============================================================
# MESH
# ============================================================

@export var max_triangle_edge: float = 2.0


# ============================================================
# VISUAL SETTINGS
# ============================================================

@export var show_mesh: bool = true

@export var show_edges: bool = false

@export var show_vertices: bool = false

@export var show_semantics: bool = true

@export var mesh_alpha: float = 0.78


# ============================================================
# CAMERA / LOOK SETTINGS
# ============================================================

@export var look_rotation_speed: float = 1.5

@export var movement_speed: float = 5.0

@export var min_zoom: float = 0.5

@export var max_zoom: float = 3.0


# ============================================================
# DATA
# ============================================================

var points: Array[Vector3] = []

var point_labels: Array[int] = []


# ============================================================
# PRECOMPUTED MESH DATA
# ============================================================

var mesh_points: Array[Vector3] = []

var mesh_labels: Array[int] = []


# ============================================================
# MESH
# ============================================================

var triangles: Array[Vector3i] = []

var edges: Array[Vector2i] = []


# ============================================================
# VIRTUAL EGO POSITION
# ============================================================

# The virtual vehicle starts at the LiDAR origin.
var ego_position: Vector3 = Vector3.ZERO


# Rotation around the Z axis.
#
# 0 degrees = looking forward along +X.
var view_yaw: float = 0.0


# ============================================================
# READY
# ============================================================

func _ready() -> void:

	print("========================================")
	print("Semantic2D")
	print("FIRST-PERSON 2.5D LiDAR VIEW")
	print("========================================")


	# --------------------------------------------------------
	# Camera2D
	# --------------------------------------------------------

	camera.position = map_center

	camera.zoom = Vector2(
		1.0,
		1.0
	)

	camera.position_smoothing_enabled = false

	camera.enabled = true


	# --------------------------------------------------------
	# Build frame list and load the first frame.
	# --------------------------------------------------------

	build_frame_sequence()
	load_trajectory_poses()
	current_frame_index = 0
	current_sequence = SEQUENCE_NAME
	current_frame = START_FRAME
	load_frame(current_sequence, current_frame)


	queue_redraw()


# ============================================================
# FRAME PATH HELPERS
# ============================================================

func get_frame_name(frame_number: int) -> String:

	return str(frame_number).pad_zeros(
		FRAME_DIGITS
	)


func get_mesh_path(sequence: String, frame_number: int) -> String:
	var frame_name: String = get_frame_name(frame_number)
	var base_dir: String = DATA_DIR + sequence + "/"

	# Prefer the correctly spelled Poisson folder/file naming.
	# Also accept the alternate "poison" spelling used by some existing exports.
	var candidates: PackedStringArray = PackedStringArray([
		base_dir + "poisson_meshes/" + frame_name + "_poisson.meshbin",
		base_dir + "poison_meshes/" + frame_name + "_poison.meshbin",
		base_dir + "poisson_meshes/" + frame_name + "_poison.meshbin",
		base_dir + "poison_meshes/" + frame_name + "_poisson.meshbin"
	])

	for candidate: String in candidates:
		if FileAccess.file_exists(candidate):
			return candidate

	# Return the standard path so the existing missing-file error is informative.
	return candidates[0]


# ============================================================
# LOAD VEHICLE TRAJECTORY POSES
# ============================================================

func load_trajectory_poses() -> void:
	trajectory_positions.clear()

	if not FileAccess.file_exists(POSES_PATH):
		push_warning("Pose file not found: " + POSES_PATH)
		return

	var file: FileAccess = FileAccess.open(POSES_PATH, FileAccess.READ)
	if file == null:
		push_warning("Could not open pose file: " + POSES_PATH)
		return

	var lines: PackedStringArray = file.get_as_text().split("\n", false)
	file.close()

	for line: String in lines:
		var trimmed: String = line.strip_edges()
		if trimmed.is_empty():
			continue

		var values: PackedStringArray = trimmed.split(" ", false)
		# Each pose is a row-major 3x4 transform (12 values).
		if values.size() != 12:
			continue

		var numbers: Array[float] = []
		var valid: bool = true
		for value: String in values:
			if not value.is_valid_float():
				valid = false
				break
			numbers.append(value.to_float())

		if not valid:
			continue

		# SemanticKITTI pose translations are stored at indices 3, 7, 11.
		# Use X/Z as the ground plane for the standard KITTI pose frame.
		trajectory_positions.append(
			Vector3(numbers[3], numbers[7], numbers[11])
		)

	print("Trajectory poses loaded: ", trajectory_positions.size())


# ============================================================
# LOAD FRAME
# ============================================================

func load_frame(sequence: String, frame_number: int) -> void:
	var mesh_path: String = get_mesh_path(sequence, frame_number)

	print("")
	print("========================================")
	print("Loading sequence ", sequence, " / frame ", get_frame_name(frame_number))
	print("Poisson mesh: ", mesh_path)
	print("========================================")

	if not FileAccess.file_exists(mesh_path):
		push_error("Precomputed Poisson mesh not found: " + mesh_path)
		return

	var sequence_changed: bool = current_sequence != sequence
	if sequence_changed:
		ego_position = Vector3.ZERO
		view_yaw = 0.0

	if not load_poisson_meshbin(mesh_path):
		push_error("Failed to load precomputed Poisson mesh.")
		return

	current_sequence = sequence
	current_frame = frame_number

	print("Mesh vertices: ", mesh_points.size())
	print("Mesh triangles: ", triangles.size())
	if not mesh_points.is_empty():
		var min_x: float = mesh_points[0].x
		var max_x: float = mesh_points[0].x
		var min_y: float = mesh_points[0].y
		var max_y: float = mesh_points[0].y
		var min_z: float = mesh_points[0].z
		var max_z: float = mesh_points[0].z
		for p: Vector3 in mesh_points:
			min_x = minf(min_x, p.x)
			max_x = maxf(max_x, p.x)
			min_y = minf(min_y, p.y)
			max_y = maxf(max_y, p.y)
			min_z = minf(min_z, p.z)
			max_z = maxf(max_z, p.z)
		print("Mesh bounds X: ", min_x, " .. ", max_x)
		print("Mesh bounds Y: ", min_y, " .. ", max_y)
		print("Mesh bounds Z: ", min_z, " .. ", max_z)
	print("========================================")
	print("Sequence ", current_sequence, " / Frame ", get_frame_name(current_frame), " loaded.")
	print("========================================")

	queue_redraw()


func load_poisson_meshbin(path: String) -> bool:
	var file: FileAccess = FileAccess.open(path, FileAccess.READ)
	if file == null:
		push_error("Could not open Poisson mesh: " + path)
		return false

	# P2DM v2 format:
	# 4-byte magic, uint32 version, uint32 vertex count, uint32 triangle count.
	# Each vertex: float32 x, y, z, int32 semantic ID.
	# Each triangle: uint32 vertex indices a, b, c.
	if file.get_length() < 16:
		file.close()
		push_error("Poisson mesh header is incomplete.")
		return false

	var magic: PackedByteArray = file.get_buffer(4)
	if magic.size() != 4 or magic.get_string_from_ascii() != "P2DM":
		file.close()
		push_error("Invalid meshbin magic. Expected P2DM.")
		return false

	var version: int = file.get_32()
	var vertex_count: int = file.get_32()
	var triangle_count: int = file.get_32()

	if version != 2:
		file.close()
		push_error("Unsupported meshbin version: " + str(version) + " (expected 2).")
		return false

	var expected_size: int = 16 + vertex_count * 16 + triangle_count * 12
	if file.get_length() != expected_size:
		file.close()
		push_error("Meshbin size does not match its header counts.")
		return false

	mesh_points.clear()
	mesh_labels.clear()
	triangles.clear()
	edges.clear()
	points.clear()
	point_labels.clear()

	for i: int in range(vertex_count):
		var x: float = file.get_float()
		var y: float = file.get_float()
		var z: float = file.get_float()
		var semantic_id: int = file.get_32()
		mesh_points.append(Vector3(x, y, z))
		mesh_labels.append(semantic_id)

	var edge_set: Dictionary = {}
	for i: int in range(triangle_count):
		var a: int = file.get_32()
		var b: int = file.get_32()
		var c: int = file.get_32()

		if a >= vertex_count or b >= vertex_count or c >= vertex_count:
			file.close()
			push_error("Meshbin contains a triangle with an invalid vertex index.")
			return false

		triangles.append(Vector3i(a, b, c))
		add_unique_edge(edge_set, a, b)
		add_unique_edge(edge_set, b, c)
		add_unique_edge(edge_set, c, a)

	file.close()

	for edge_key: Variant in edge_set:
		var parts: PackedStringArray = str(edge_key).split("_")
		if parts.size() == 2:
			edges.append(Vector2i(int(parts[0]), int(parts[1])))

	return true


func add_unique_edge(edge_set: Dictionary, a: int, b: int) -> void:
	var first: int = mini(a, b)
	var second: int = maxi(a, b)
	edge_set[str(first) + "_" + str(second)] = true


# ============================================================
# WORLD → CAMERA SPACE
# ============================================================

func transform_to_camera_space(
	world_point: Vector3
) -> Vector3:

	var relative: Vector3 = (
		world_point
		- ego_position
	)


	var cos_yaw: float = cos(
		view_yaw
	)

	var sin_yaw: float = sin(
		view_yaw
	)


	var camera_x: float = (
		relative.x * cos_yaw
		+ relative.y * sin_yaw
	)


	var camera_y: float = (
		-relative.x * sin_yaw
		+ relative.y * cos_yaw
	)


	var camera_z: float = (
		relative.z
		- camera_height
	)


	return Vector3(
		camera_x,
		camera_y,
		camera_z
	)


# ============================================================
# PERSPECTIVE PROJECTION
# ============================================================

func project_point(
	world_point: Vector3
) -> Vector2:

	var camera_point: Vector3 = (
		transform_to_camera_space(
			world_point
		)
	)


	var depth: float = (
		camera_point.x
	)


	if depth < near_clip:

		depth = near_clip


	var perspective_x: float = (
		camera_point.y
		* focal_length
		/ depth
	)


	var perspective_y: float = (
		-camera_point.z
		* focal_length
		/ depth
	)


	return (
		map_center
		+ Vector2(
			perspective_x,
			perspective_y
		)
	)


# ============================================================
# CHECK IF POINT IS VISIBLE
# ============================================================

func is_point_visible(
	world_point: Vector3
) -> bool:

	var camera_point: Vector3 = (
		transform_to_camera_space(
			world_point
		)
	)


	return (
		camera_point.x >= near_clip
		and camera_point.x <= far_clip
	)


# ============================================================
# SEMANTIC COLOR
# ============================================================

func get_semantic_color(
	semantic_id: int
) -> Color:

	match semantic_id:

		0:
			return Color(
				0.35,
				0.35,
				0.35
			)

		1:
			return Color(
				0.55,
				0.55,
				0.55
			)

		10:
			return Color(
				0.10,
				0.80,
				0.95
			)

		11:
			return Color(
				0.20,
				0.70,
				1.00
			)

		13:
			return Color(
				1.00,
				0.35,
				0.10
			)

		15:
			return Color(
				0.90,
				0.20,
				0.80
			)

		16:
			return Color(
				0.75,
				0.20,
				0.90
			)

		18:
			return Color(
				0.95,
				0.55,
				0.10
			)

		20:
			return Color(
				0.90,
				0.45,
				0.20
			)

		30:
			return Color(
				1.00,
				0.20,
				0.20
			)

		31:
			return Color(
				1.00,
				0.50,
				0.20
			)

		32:
			return Color(
				1.00,
				0.30,
				0.50
			)

		40:
			return Color(
				0.20,
				0.20,
				0.25
			)

		44:
			return Color(
				0.30,
				0.40,
				0.45
			)

		48:
			return Color(
				0.55,
				0.55,
				0.55
			)

		49:
			return Color(
				0.40,
				0.45,
				0.40
			)

		50:
			return Color(
				0.70,
				0.45,
				0.30
			)

		51:
			return Color(
				0.75,
				0.65,
				0.20
			)

		52:
			return Color(
				0.50,
				0.40,
				0.35
			)

		60:
			return Color(
				1.00,
				1.00,
				0.20
			)

		70:
			return Color(
				0.15,
				0.75,
				0.20
			)

		71:
			return Color(
				0.45,
				0.25,
				0.10
			)

		72:
			return Color(
				0.50,
				0.65,
				0.30
			)

		80:
			return Color(
				0.70,
				0.70,
				0.75
			)

		81:
			return Color(
				1.00,
				0.15,
				0.15
			)

		_:
			return Color(
				0.30,
				0.30,
				0.80
			)


# ============================================================
# DRAW
# ============================================================

func _draw() -> void:

	# --------------------------------------------------------
	# BACKGROUND
	# --------------------------------------------------------

	# Match the background to the active viewport, accounting for Camera2D.
	var viewport_size: Vector2 = get_viewport_rect().size
	var visible_world_rect := Rect2(
		camera.position - viewport_size * 0.5,
		viewport_size
	)
	draw_rect(
		visible_world_rect,
		Color(0.025, 0.025, 0.035)
	)


	# --------------------------------------------------------
	# HORIZON
	# --------------------------------------------------------

	# Horizon spans the current viewport rather than a hard-coded width.
	var horizon_y: float = camera.position.y - viewport_size.y * 0.5 + viewport_size.y * 0.36
	draw_line(
		Vector2(visible_world_rect.position.x, horizon_y),
		Vector2(visible_world_rect.end.x, horizon_y),
		Color(0.12, 0.12, 0.15, 0.35),
		1.0
	)


	# --------------------------------------------------------
	# DRAW TRIANGLES
	# --------------------------------------------------------

	var visible_triangle_count: int = 0
	var visible_vertex_count: int = 0

	# Cache projected positions and visibility once per vertex. This preserves
	# full detail while reducing repeated work during mouse movement.
	var projected_points: Array[Vector2] = []
	var point_is_visible: Array[bool] = []
	projected_points.resize(mesh_points.size())
	point_is_visible.resize(mesh_points.size())
	for i: int in range(mesh_points.size()):
		var camera_point: Vector3 = transform_to_camera_space(mesh_points[i])
		point_is_visible[i] = (
			camera_point.x >= near_clip
			and camera_point.x <= far_clip
		)
		if point_is_visible[i]:
			projected_points[i] = project_point(mesh_points[i])

	if show_mesh:
		# Draw every triangle during dragging and when idle.
		for tri_index: int in range(triangles.size()):
			var tri: Vector3i = triangles[tri_index]

			if tri.x >= mesh_points.size():
				continue

			if tri.y >= mesh_points.size():
				continue

			if tri.z >= mesh_points.size():
				continue


			var world_a: Vector3 = (
				mesh_points[tri.x]
			)

			var world_b: Vector3 = (
				mesh_points[tri.y]
			)

			var world_c: Vector3 = (
				mesh_points[tri.z]
			)


			if not point_is_visible[tri.x]:
				continue
			if not point_is_visible[tri.y]:
				continue
			if not point_is_visible[tri.z]:
				continue

			var p1: Vector2 = projected_points[tri.x]
			var p2: Vector2 = projected_points[tri.y]
			var p3: Vector2 = projected_points[tri.z]


			if p1.distance_to(p2) > 500.0:
				continue

			if p2.distance_to(p3) > 500.0:
				continue

			if p3.distance_to(p1) > 500.0:
				continue


			var label_a: int = mesh_labels[tri.x]
			var label_b: int = mesh_labels[tri.y]
			var label_c: int = mesh_labels[tri.z]
			var semantic_id: int = label_a
			if label_b == label_c or label_b == label_a:
				semantic_id = label_b
			elif label_c == label_a:
				semantic_id = label_c


			var triangle_color: Color


			if show_semantics:

				triangle_color = (
					get_semantic_color(
						semantic_id
					)
				)

			else:

				triangle_color = Color(
					0.20,
					0.70,
					1.00
				)


			triangle_color.a = mesh_alpha


			draw_colored_polygon(
				PackedVector2Array([
					p1,
					p2,
					p3
				]),
				triangle_color
			)
			visible_triangle_count += 1

	# If no faces survive clipping/projection, draw visible mesh vertices as a
	# diagnostic fallback so a valid mesh does not leave the view completely blank.
	if show_mesh and visible_triangle_count == 0:
		for i: int in range(mesh_points.size()):
			var point: Vector3 = mesh_points[i]
			if not is_point_visible(point):
				continue
			var screen_position: Vector2 = project_point(point)
			if screen_position.x < -100.0 or screen_position.x > get_viewport_rect().size.x + 100.0:
				continue
			if screen_position.y < -100.0 or screen_position.y > get_viewport_rect().size.y + 100.0:
				continue
			draw_circle(screen_position, 2.0, get_semantic_color(mesh_labels[i]))
			visible_vertex_count += 1

	# --------------------------------------------------------
	# OPTIONAL EDGES
	# --------------------------------------------------------

	if show_edges:

		for edge: Vector2i in edges:

			if edge.x >= mesh_points.size():
				continue

			if edge.y >= mesh_points.size():
				continue


			var a: Vector3 = (
				mesh_points[edge.x]
			)

			var b: Vector3 = (
				mesh_points[edge.y]
			)


			if not is_point_visible(a):
				continue

			if not is_point_visible(b):
				continue


			var p1: Vector2 = (
				project_point(a)
			)

			var p2: Vector2 = (
				project_point(b)
			)


			if p1.distance_to(p2) > 500.0:
				continue


			draw_line(
				p1,
				p2,
				Color(
					0.85,
					0.85,
					0.85,
					0.25
				),
				0.7
			)


	# --------------------------------------------------------
	# OPTIONAL POINTS
	# --------------------------------------------------------

	if show_vertices:

		for i: int in range(
			mesh_points.size()
		):

			var point: Vector3 = (
				mesh_points[i]
			)


			if not is_point_visible(point):
				continue


			var screen_position: Vector2 = (
				project_point(point)
			)


			var distance: float = (
				transform_to_camera_space(
					point
				).x
			)


			var radius: float = clampf(
				20.0 / max(distance, 1.0),
				1.0,
				5.0
			)


			draw_circle(
				screen_position,
				radius,
				get_semantic_color(
					mesh_labels[i]
				)
			)


	# --------------------------------------------------------
	# EGO VEHICLE / HUD
	# --------------------------------------------------------

	draw_ego_view()


	# --------------------------------------------------------
	# SEQUENCE / FRAME
	# --------------------------------------------------------

	draw_string(
		ThemeDB.fallback_font,
		Vector2(
			20.0,
			30.0
		),
		"Sequence: " + current_sequence,
		HORIZONTAL_ALIGNMENT_LEFT,
		-1,
		18,
		Color.WHITE
	)

	draw_string(
		ThemeDB.fallback_font,
		Vector2(
			20.0,
			55.0
		),
		"Frame: " + get_frame_name(
			current_frame
		),
		HORIZONTAL_ALIGNMENT_LEFT,
		-1,
		18,
		Color.WHITE
	)

	draw_string(
		ThemeDB.fallback_font,
		Vector2(20.0, 80.0),
		"Mesh: %d vertices | %d triangles | %d faces drawn | %d fallback points" % [mesh_points.size(), triangles.size(), visible_triangle_count, visible_vertex_count],
		HORIZONTAL_ALIGNMENT_LEFT,
		-1,
		14,
		Color(0.75, 0.9, 1.0)
	)

	# Top-down route overview from the recorded pose file.
	if show_trajectory:
		draw_trajectory_minimap()


# ============================================================
# TRAJECTORY MINIMAP
# ============================================================

func draw_trajectory_minimap() -> void:
	if trajectory_positions.size() < 2:
		return

	var viewport_size: Vector2 = get_viewport_rect().size
	var panel_position: Vector2 = Vector2(
		camera.position.x + viewport_size.x * 0.5 - trajectory_panel_size.x - 20.0,
		camera.position.y - viewport_size.y * 0.5 + 18.0
	)
	var panel: Rect2 = Rect2(panel_position, trajectory_panel_size)
	draw_rect(panel, Color(0.025, 0.035, 0.05, 0.88), true)
	draw_rect(panel, Color(0.35, 0.45, 0.55, 0.9), false, 1.0)

	draw_string(
		ThemeDB.fallback_font,
		panel_position + Vector2(10.0, 19.0),
		"VEHICLE TRAJECTORY",
		HORIZONTAL_ALIGNMENT_LEFT,
		-1,
		12,
		Color(0.85, 0.9, 0.95)
	)

	var plot_rect: Rect2 = Rect2(
		panel_position + Vector2(trajectory_padding, 28.0),
		trajectory_panel_size - Vector2(trajectory_padding * 2.0, 40.0)
	)

	var min_x: float = trajectory_positions[0].x
	var max_x: float = min_x
	var min_z: float = trajectory_positions[0].z
	var max_z: float = min_z

	for position: Vector3 in trajectory_positions:
		min_x = minf(min_x, position.x)
		max_x = maxf(max_x, position.x)
		min_z = minf(min_z, position.z)
		max_z = maxf(max_z, position.z)

	var span_x: float = maxf(max_x - min_x, 0.001)
	var span_z: float = maxf(max_z - min_z, 0.001)
	var scale_factor: float = minf(plot_rect.size.x / span_x, plot_rect.size.y / span_z)
	var drawn_size: Vector2 = Vector2(span_x, span_z) * scale_factor
	var offset: Vector2 = plot_rect.position + (plot_rect.size - drawn_size) * 0.5

	# Convert the recorded X/Z ground-plane coordinates into minimap pixels.
	var screen_positions: Array[Vector2] = []
	for position: Vector3 in trajectory_positions:
		var screen_position: Vector2 = Vector2(
			offset.x + (position.x - min_x) * scale_factor,
			offset.y + (max_z - position.z) * scale_factor
		)
		screen_positions.append(screen_position)

	# Full route in a subdued color.
	for i: int in range(1, screen_positions.size()):
		draw_line(
			screen_positions[i - 1],
			screen_positions[i],
			Color(0.25, 0.45, 0.58, 0.65),
			1.5
		)

	# Highlight the route travelled up to the currently displayed frame.
	var last_index: int = mini(current_frame, screen_positions.size() - 1)
	for i: int in range(1, last_index + 1):
		draw_line(
			screen_positions[i - 1],
			screen_positions[i],
			Color(0.0, 0.85, 1.0, 1.0),
			2.2
		)

	# Start marker.
	draw_circle(screen_positions[0], 3.0, Color(0.3, 1.0, 0.45, 1.0))

	# Current vehicle position.
	var current_position: Vector2 = screen_positions[last_index]
	draw_circle(current_position, 4.5, Color(1.0, 0.85, 0.15, 1.0))
	draw_circle(current_position, 2.0, Color(1.0, 1.0, 1.0, 1.0))

	draw_string(
		ThemeDB.fallback_font,
		panel_position + Vector2(10.0, trajectory_panel_size.y - 8.0),
		"START  •  cyan: travelled  •  yellow: current",
		HORIZONTAL_ALIGNMENT_LEFT,
		-1,
		10,
		Color(0.8, 0.85, 0.9)
	)


# ============================================================
# EGO VEHICLE
# ============================================================

func draw_ego_view() -> void:

	var center: Vector2 = (
		Vector2(
			map_center.x,
			580.0
		)
	)


	var body: PackedVector2Array = (
		PackedVector2Array([
			center + Vector2(-24.0, 25.0),
			center + Vector2(24.0, 25.0),
			center + Vector2(20.0, -15.0),
			center + Vector2(0.0, -35.0),
			center + Vector2(-20.0, -15.0)
		])
	)


	draw_colored_polygon(
		body,
		Color(
			0.05,
			0.65,
			1.00,
			0.95
		)
	)


	for i: int in range(
		body.size()
	):

		var next_index: int = (
			(i + 1)
			% body.size()
		)


		draw_line(
			body[i],
			body[next_index],
			Color.WHITE,
			2.0
		)


	draw_line(
		center,
		center + Vector2(
			0.0,
			-70.0
		),
		Color(
			1.0,
			0.85,
			0.1,
			0.8
		),
		2.0
	)


	draw_circle(
		Vector2(
			map_center.x,
			map_center.y
		),
		3.0,
		Color.WHITE
	)


	draw_line(
		Vector2(
			map_center.x - 12.0,
			map_center.y
		),
		Vector2(
			map_center.x + 12.0,
			map_center.y
		),
		Color(
			1.0,
			1.0,
			1.0,
			0.45
		),
		1.0
	)


	draw_line(
		Vector2(
			map_center.x,
			map_center.y - 12.0
		),
		Vector2(
			map_center.x,
			map_center.y + 12.0
		),
		Color(
			1.0,
			1.0,
			1.0,
			0.45
		),
		1.0
	)


# ============================================================
# INPUT / MOVEMENT
# ============================================================

func _process(delta: float) -> void:

	if playback_paused:
		return

	playback_timer += delta

	if playback_timer >= maxf(frame_interval, 0.05):
		playback_timer = 0.0
		advance_playback()


func advance_playback() -> void:

	var next_index: int = current_frame_index + 1

	if next_index >= FRAME_SEQUENCE.size():
		playback_paused = true
		print("End of configured frame list at sequence ", current_sequence,
			" / frame ", get_frame_name(current_frame))
		return

	var next_entry: Dictionary = FRAME_SEQUENCE[next_index]
	var next_sequence: String = str(next_entry["sequence"])
	var next_frame: int = int(next_entry["frame"])

	# Do not treat separate SemanticKITTI sequences as one continuous drive.
	if next_sequence != current_sequence:
		playback_paused = true
		print("End of sequence ", current_sequence,
			" at frame ", get_frame_name(current_frame),
			". Playback paused before sequence ", next_sequence, ".")
		return

	current_frame_index = next_index
	load_frame(next_sequence, next_frame)


# ============================================================
# KEYBOARD / MOUSE INPUT
# ============================================================

func _unhandled_input(event: InputEvent) -> void:

	# Spacebar toggles automatic playback.
	if event is InputEventKey:
		var key_event: InputEventKey = event

		if key_event.pressed and not key_event.echo:
			if key_event.keycode == KEY_SPACE:
				playback_paused = not playback_paused
				playback_timer = 0.0
				mouse_dragging = false

				if playback_paused:
					print("Playback paused at sequence ", current_sequence,
						" / frame ", get_frame_name(current_frame))
				else:
					print("Playback resumed at sequence ", current_sequence,
						" / frame ", get_frame_name(current_frame))

				get_viewport().set_input_as_handled()
				return

			# Keep the original reset shortcut.
			elif key_event.keycode == KEY_R:
				playback_timer = 0.0
				current_frame_index = 0

				var first_entry: Dictionary = FRAME_SEQUENCE[0]
				current_sequence = str(first_entry["sequence"])
				current_frame = int(first_entry["frame"])

				ego_position = Vector3.ZERO
				view_yaw = 0.0
				camera_height = 1.5
				camera.position = map_center
				camera.zoom = Vector2(1.0, 1.0)

				load_frame(current_sequence, current_frame)
				queue_redraw()
				get_viewport().set_input_as_handled()
				return

	# Mouse inspection is enabled while playback is paused.
	if event is InputEventMouseButton:
		var mouse_event: InputEventMouseButton = event

		if mouse_event.button_index == MOUSE_BUTTON_LEFT:
			if playback_paused:
				mouse_dragging = mouse_event.pressed
				last_mouse_position = mouse_event.position
				queue_redraw()
				get_viewport().set_input_as_handled()
			return

		# Preserve the original mouse-wheel zoom controls.
		if mouse_event.pressed and mouse_event.button_index == MOUSE_BUTTON_WHEEL_UP:
			camera.zoom *= Vector2(1.15, 1.15)
			clamp_zoom()
			return

		if mouse_event.pressed and mouse_event.button_index == MOUSE_BUTTON_WHEEL_DOWN:
			camera.zoom *= Vector2(0.85, 0.85)
			clamp_zoom()
			return

	if event is InputEventMouseMotion and playback_paused and mouse_dragging:
		var mouse_motion: InputEventMouseMotion = event
		var mouse_delta: Vector2 = mouse_motion.position - last_mouse_position
		last_mouse_position = mouse_motion.position

		# Horizontal drag rotates the existing 2.5D viewpoint.
		view_yaw -= mouse_delta.x * mouse_look_sensitivity

		# Vertical drag adjusts the virtual camera height for a 2.5D look angle.
		camera_height = clampf(
			camera_height + mouse_delta.y * mouse_height_sensitivity,
			0.2,
			8.0
		)

		queue_redraw()
		get_viewport().set_input_as_handled()


# ============================================================
# CLAMP ZOOM
# ============================================================

func clamp_zoom() -> void:

	var value: float = (
		camera.zoom.x
	)


	value = clampf(
		value,
		min_zoom,
		max_zoom
	)


	camera.zoom = Vector2(
		value,
		value
	)
