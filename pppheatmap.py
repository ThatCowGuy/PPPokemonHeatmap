
# https://zulko.github.io/moviepy/ref/videofx/moviepy.video.fx.all.crop.html?highlight=crop
import moviepy.editor as mp_editor
import moviepy.video as mp_video

import os
import cv2
import numpy as np
  
stream = mp_editor.VideoFileClip("stream1.mp4")

# weird fix for subclip having issues
print(stream.duration)
stream = stream.set_duration(stream.duration)
print(stream.duration)

clip_start = 14*60*60 + 38*60
clip_len = 15*60
target_FPS = 3
video_mode = True
clip_start_frame = clip_start * target_FPS

# temporarily, for testing, trim the clip A LOT
stream = stream.subclip(t_start=clip_start, t_end=clip_start+clip_len)

# strip audio
stream = stream.without_audio()
# crop viewport
stream = mp_video.fx.all.crop(stream, x1=168, x2=273, y1=45, y2=115)

# lower FPS (instead of 60 FPS, only keep a couple of frames)
stream = stream.set_fps(target_FPS)
frame_cnt = clip_len * target_FPS
print(f"Condensed Stream into {frame_cnt} Frames")
print(f"Duration: {stream.duration}, Dimensions: {stream.size}")

# stream.write_videofile("cut.mp4")
if (video_mode == True):
	for idx, frame in enumerate(stream.iter_frames()):
		cv2.imwrite(f"frames/frame{(clip_start_frame + idx):06d}.png", cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY))

# list of searchable maps (each map also knows which maps it's connected to, to reduce search space)
# format: map_name, loaded_img_file, connected_map_names
searchable_maps = [
	("littleroot", 		cv2.cvtColor(cv2.imread(f"lowres_maps/littleroot.png"), cv2.COLOR_BGR2GRAY),	["seths_home", "mays_home", "lab", "route101"]),
	("seths_bedroom", 	cv2.cvtColor(cv2.imread(f"lowres_maps/seths_bedroom.png"), cv2.COLOR_BGR2GRAY),	["seths_home"]),
	("seths_home", 		cv2.cvtColor(cv2.imread(f"lowres_maps/seths_home.png"), cv2.COLOR_BGR2GRAY),	["seths_bedroom", "littleroot"]),
	("mays_bedroom", 	cv2.cvtColor(cv2.imread(f"lowres_maps/mays_bedroom.png"), cv2.COLOR_BGR2GRAY),	["mays_home"]),
	("mays_home", 		cv2.cvtColor(cv2.imread(f"lowres_maps/mays_home.png"), cv2.COLOR_BGR2GRAY),		["mays_bedroom", "littleroot"]),
	("lab", 			cv2.cvtColor(cv2.imread(f"lowres_maps/lab.png"), cv2.COLOR_BGR2GRAY),			["littleroot"]),
	("route101", 		cv2.cvtColor(cv2.imread(f"lowres_maps/route101.png"), cv2.COLOR_BGR2GRAY),		["littleroot"]),
]

pause_menus = [ ]
for file in os.listdir("menu_templates"):
	filename = os.fsdecode(file)
	if (filename.endswith(".png")):
		pause_menus.append((filename[:-4], cv2.cvtColor(cv2.imread(f"menu_templates/{filename}"), cv2.COLOR_BGR2GRAY)))
def check_for_menus(frame):
	# check for all the menu-templates that I made
	for pause_menu in pause_menus:
		# do the matching
		match_matrix = cv2.matchTemplate(frame, pause_menu[1], cv2.TM_CCORR_NORMED, mask=pause_menu[1])
		__, max_val, __, __ = cv2.minMaxLoc(match_matrix)
		if (max_val >= 0.995):
			return pause_menu[0]
	# check if it is a very dark or even black screen
	if (np.mean(frame) < 20):
		return "blackscreen"
	return None

# to switch to another map later on
def get_map_by_name(map_name):
	for map in searchable_maps:
		if (map[0] == map_name):
			return map
	return None

# every map gets its own heatmap
heatmaps = [] * len(searchable_maps)

threshold_A = 0.98 # slightly favor the current map
threshold_B = 0.98
px_size = 7
x_offset = 7.50
y_offset = 4.95

seth_mask = cv2.imread("seth_mask.png")
seth_mask = cv2.cvtColor(seth_mask, cv2.COLOR_BGR2GRAY)
seth_mask[seth_mask > 0] = 255

# find the inital map by scanning ALL of the available ones
# (Note: first couple frames might be menuing, so we need to extend this search a bit)
current_map = None
for idx, frame in enumerate(stream.iter_frames()):
	viewport_img = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
	for map in searchable_maps:
		map_img = cv2.imread(f"lowres_maps/{map[0]}.png")
		map_img = cv2.cvtColor(map_img, cv2.COLOR_BGR2GRAY)
		# do the matching
		match_matrix = cv2.matchTemplate(map[1], viewport_img, cv2.TM_CCORR_NORMED, mask=seth_mask)
		__, max_val, __, max_loc = cv2.minMaxLoc(match_matrix)
		# check the threshold
		if (max_val < threshold_B):
			continue
		# if we got up to here, we found the map !
		current_map = map
		last_map = current_map
		print(f"intial map is: {current_map[0]} (on frame {(clip_start_frame + idx):06d})")
		break
	if (current_map is not None):
		break

last_seth_x = -9999
last_seth_y = -9999
last_x_px = -9999
last_y_px = -9999
last_map_name = current_map[0]
map_ROI_x = 0
map_ROI_y = 0
ROI_is_updated = False
menu_cnt = 0
dupe_cnt = 0
move_cnt = 0

# creating an empty image as a starting point
last_checked_img = np.zeros((stream.size[1], stream.size[0]), np.uint8)
if (video_mode == False):
	# now do the actual, restricted search for the entire clip len
	for idx, frame in enumerate(stream.iter_frames()):
		viewport_img = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

		# check if the image changed at all first
		match_matrix = cv2.matchTemplate(last_checked_img, viewport_img, cv2.TM_CCORR_NORMED)
		last_checked_img = viewport_img
		if (cv2.minMaxLoc(match_matrix)[1] > 0.995):
			dupe_cnt += 1
			continue
		# check if the image is just some menu
		potential_menu = check_for_menus(viewport_img)
		if (potential_menu is not None):
			# print(f"F-{(clip_start_frame + idx):06d} (-): Menu: {potential_menu}")
			menu_cnt += 1
			continue
		
		# search through the current map, as well as all the maps connected to it
		for map_name in [current_map[0], *current_map[2]]:
			# if we haven't seen this scene yet, dont restrict the ROI
			if (map_name != last_map_name or last_x_px == -9999 or last_y_px == -9999):
				map_ROI_x = 0
				map_ROI_y = 0
				map_ROI = get_map_by_name(map_name)[1]
				ROI_is_updated = False
			# otherwise (if the current ROI is not updated), constrain the search space by A LOT
			elif (ROI_is_updated == False):
				map_ROI_x = int(last_x_px - 2.25*px_size)
				map_ROI_y = int(last_y_px - 2.25*px_size)
				map_ROI_x = 0 if map_ROI_x <= 0 else map_ROI_x
				map_ROI_y = 0 if map_ROI_y <= 0 else map_ROI_y
				if (map_ROI_x == 0 or map_ROI_y == 0):
					print(f"[-] ROI landed in weird place !! ROI={map_ROI_x},{map_ROI_y} and lastpx={last_x_px},{last_y_px}")
				map_ROI = get_map_by_name(map_name)[1][
					map_ROI_y : int(map_ROI_y + 14.5*px_size),
					map_ROI_x : int(map_ROI_x + 19.5*px_size)
				]
				ROI_is_updated = True

			# do the matching
			match_matrix = cv2.matchTemplate(map_ROI, viewport_img, cv2.TM_CCORR_NORMED, mask=seth_mask)
			__, max_val, __, max_loc = cv2.minMaxLoc(match_matrix)
			# check the threshold
			threshold = threshold_A if map_name == last_map_name else threshold_B
			if (max_val < threshold):
				continue

			# extract the best match
			match_x = max_loc[0] + map_ROI_x
			match_y = max_loc[1] + map_ROI_y
			
			# before doing any math, check if the detection-px changed at all
			if (match_x == last_x_px and match_y == last_y_px and map_name == last_map_name):
				continue
			last_x_px = match_x
			last_y_px = match_y

			# get seth's x and y coords;
			# seth should be x_offset tiles away from the left border of the detection,
			# and roughly y_offset tiles away from the top border.
			seth_x_px = int(match_x + (x_offset * px_size))
			seth_y_px = int(match_y + (y_offset * px_size))
			# from this pixel location I can get the box ID by integer-dividing by px-size (Note: 0-indexing)
			seth_x = int((seth_x_px - 1) // px_size)
			seth_y = int((seth_y_px - 1) // px_size)

			# check if seth moved at all
			if (seth_x == last_seth_x and seth_y == last_seth_y and map_name == last_map_name):
				continue
			print(f"F-{(clip_start_frame + idx):06d} ({map_name}): Seth is at ({seth_x}, {seth_y})")
			ROI_is_updated = False
			move_cnt += 1

			# check if seth moved an obscene distance
			if (map_name == last_map_name and last_seth_x != -9999):
				x_diff = abs(seth_x - last_seth_x)
				y_diff = abs(seth_y - last_seth_y)
				if (x_diff > 0 and y_diff > 0):
					print("DIAGONAL MOVEMENT !?")
					exit(0)
				if (x_diff > 2 or y_diff > 2):
					print("MOVEMENT OF >3 TILES !?")
					exit(0)
				if (x_diff == 2):
					print("double step X")
				elif (y_diff == 2):
					print("double step Y")
			
			last_seth_x = seth_x
			last_seth_y = seth_y
			last_map_name = map_name
			current_map = get_map_by_name(map_name)
			break

if (video_mode == True):
	# now do the actual, restricted search for the entire clip len
	for idx, frame in enumerate(stream.iter_frames()):
		viewport_img = cv2.imread(f"frames/frame{(clip_start_frame + idx):06d}.png")
		viewport_img = cv2.cvtColor(viewport_img, cv2.COLOR_BGR2GRAY)
		# search through the current map, as well as all the maps connected to it
		for map_name in [current_map[0], *current_map[2]]:
			map_img = cv2.imread(f"lowres_maps/{map_name}.png")
			map_img = cv2.cvtColor(map_img, cv2.COLOR_BGR2GRAY)
			h, w = viewport_img.shape[:]
			# do the matching
			match_matrix = cv2.matchTemplate(map_img, viewport_img, cv2.TM_CCORR_NORMED, mask=seth_mask)
			__, max_val, __, max_loc = cv2.minMaxLoc(match_matrix)
			# check the threshold
			threshold = threshold_A if map_name == last_map_name else threshold_B
			if (max_val < threshold):
				continue
			# extract the best match
			match_x = max_loc[0]
			match_y = max_loc[1]
			last_x_px = match_x
			last_y_px = match_y
			# get seth's x and y coords;
			# seth should be x_offset tiles away from the left border of the detection,
			# and roughly y_offset tiles away from the top border.
			seth_x_px = int(match_x + (x_offset * px_size))
			seth_y_px = int(match_y + (y_offset * px_size))
			# from this pixel location I can get the box ID by integer-dividing by px-size (Note: 0-indexing)
			seth_x = int((seth_x_px - 1) // px_size)
			seth_y = int((seth_y_px - 1) // px_size)

			# check if seth moved at all
			if (seth_x == last_seth_x and seth_y == last_seth_y and map_name == last_map_name):
				continue
			last_seth_x = seth_x
			last_seth_y = seth_y
			last_map_name = map_name
			current_map = get_map_by_name(map_name)

			# check if seth is mid movement by seeing how well the observation aligns to the grid (Note: 1-indexing)
			x_dev = np.power(((seth_x + 0.5) * px_size) + 1 - (seth_x_px), 2.0)
			y_dev = np.power(((seth_y + 0.5) * px_size) + 1 - (seth_y_px), 2.0)
			dev = np.sqrt(x_dev + y_dev)
			# realign seth regardless of deviation (because it wont matter if the match is good anyways)
			seth_x_px = int((seth_x + 0.5) * px_size)
			seth_y_px = int((seth_y + 0.5) * px_size)
			match_x = int(seth_x_px - (x_offset * px_size) + 1)
			match_y = int(seth_y_px - (y_offset * px_size) + 1)

			seth_box_x = int(seth_x * px_size)
			seth_box_y = int(seth_y * px_size)

			# and draw some visuals
			cv2.rectangle(map_img, (match_x, match_y), ((match_x + w), (match_y + h)), (255, 0, 0), 1)
			cv2.rectangle(map_img, (seth_box_x, seth_box_y), ((seth_box_x + px_size), (seth_box_y + px_size)), (0, 0, 255), 1)
			print(f"F-{(clip_start_frame + idx):06d} ({map_name}): Seth is at ({seth_x}, {seth_y})")

			h, w = map_img.shape[:]
			map_img = cv2.resize(map_img, (w*4, h*4), interpolation= cv2.INTER_LINEAR)
			cv2.imwrite(f"detections/detection{(clip_start_frame + idx):06d}.png", map_img)
			break

print(f"Filtered out {dupe_cnt} ({100*dupe_cnt/frame_cnt:.2f}%) Dupe-Frames")
print(f"Filtered out {menu_cnt} ({100*menu_cnt/frame_cnt:.2f}%) Menu-Frames.")
print(f"Important Frames detected: {move_cnt} ({100*move_cnt/frame_cnt:.2f}%) Frames.")
other_cnt = frame_cnt - (dupe_cnt + menu_cnt + move_cnt)
print(f"Unaccounted Frames: {other_cnt} ({100*other_cnt/frame_cnt:.2f}%) Frames.")