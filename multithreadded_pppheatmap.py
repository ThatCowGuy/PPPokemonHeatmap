
# https://zulko.github.io/moviepy/ref/videofx/moviepy.video.fx.all.crop.html?highlight=crop
import moviepy.editor as mp_editor
import moviepy.video as mp_video

import os
import cv2
import numpy as np
import threading
import time

target_FPS = 3
video_mode = False

threshold_A = 0.98 # for the current map
threshold_B = 0.98 # for connected maps
px_size = 7
x_offset = 7.50
y_offset = 4.95

def frame_to_timestamp(frame_idx, FPS):
	return time.strftime('%H:%M:%S', time.gmtime(frame_idx // FPS))

# depending on the proc.factor (how many stream seconds can I process in 1 second ?),
# calculate the optimal thread-optimization
def thread_optimization(thread_cnt, thread_ID, processing_factor=1):
	if (thread_cnt == 1):
		return 0
	# very minor optimization, but because it takes ~2s to pre-process the stream for each thread,
	# I can make use of this delay by making the first threads do more work
	return int(2 * processing_factor * ((thread_cnt - 1) / 2 - thread_ID))

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
seth_mask = cv2.imread("seth_mask.png")
seth_mask = cv2.cvtColor(seth_mask, cv2.COLOR_BGR2GRAY)
seth_mask[seth_mask > 0] = 255

pause_menus = [ ]
for file in os.listdir("menu_templates"):
	filename = os.fsdecode(file)
	if (filename.endswith(".png")):
		menu_img = cv2.cvtColor(cv2.imread(f"menu_templates/{filename}"), cv2.COLOR_BGR2GRAY)
		pause_menus.append((filename[:-4], menu_img, menu_img))

def check_for_menus(frame):
	# check for all the menu-templates that I made
	for pause_menu in pause_menus:
		# do the matching
		match_matrix = cv2.matchTemplate(frame, pause_menu[1], cv2.TM_CCORR_NORMED, mask=pause_menu[2])
		# this next line could easily solve a bug that happens when either the IMG or the NEEDLE are masked to be full-zero
		# but the runtime-cost is so high, that Id rather just disregard the match
		# match_matrix[match_matrix == float('inf')] = 0 # bug https://github.com/opencv/opencv/issues/15768 
		__, max_val, __, __ = cv2.minMaxLoc(match_matrix)
		if (max_val >= 1): # throwing out matches that skyrocket due to div-by-close-to-zero, or straight up inf-results
			continue
		if (max_val >= 0.999):
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



def process_stream_segment(ID, clip_start_frame):
	# find the inital map by scanning ALL of the available ones
	# (Note: first couple frames might be menuing, so we need to extend this search a bit)
	current_map = None
	for idx, frame in enumerate(trims[ID].iter_frames()):
		viewport_img = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
		for map in searchable_maps:
			# do the matching
			match_matrix = cv2.matchTemplate(map[1], viewport_img, cv2.TM_CCORR_NORMED, mask=seth_mask)
			__, max_val, __, max_loc = cv2.minMaxLoc(match_matrix)
			# check the threshold
			if (max_val < threshold_B):
				continue
			# if we got up to here, we found the map !
			current_map = map
			last_map = current_map
			frame_idx = clip_start_frame + idx
			print(f"[THREAD #{ID}] -- Intial map is: {current_map[0]} (on frame {frame_idx:06d}, @{frame_to_timestamp(frame_idx, target_FPS)})")
			break
		if (current_map is not None):
			break
	
	# init some variables that are running along with the loop
	last_seth_x = -9999
	last_seth_y = -9999
	thread_arr_seth_path[ID] = []
	last_x_px = -9999
	last_y_px = -9999
	last_map_name = current_map[0]
	map_ROI_x = 0
	map_ROI_y = 0
	ROI_is_updated = False
	menu_cnt = 0
	dupe_cnt = 0
	move_cnt = 0
	error_cnt = 0

	# creating an empty image as a starting point
	last_checked_img = np.zeros((trims[ID].size[1], trims[ID].size[0]), np.uint8)
	
	# now do the actual, restricted search for the entire clip len
	for idx, frame in enumerate(trims[ID].iter_frames()):
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

			# check if seth moved an obscene distance
			if (map_name == last_map_name and last_seth_x != -9999):
				x_diff = abs(seth_x - last_seth_x)
				y_diff = abs(seth_y - last_seth_y)
				if (x_diff > 0 and y_diff > 0):
					frame_idx = clip_start_frame + idx
					print(f"[THREAD #{ID}] -- Diagonal Movement detected (on frame {frame_idx:06d}, @{frame_to_timestamp(frame_idx, target_FPS)})")
					error_cnt += 1
				if (x_diff > 2 or y_diff > 2):
					frame_idx = clip_start_frame + idx
					print(f"[THREAD #{ID}] -- 3-Step Movement detected (on frame {frame_idx:06d}, @{frame_to_timestamp(frame_idx, target_FPS)})")
					error_cnt += 1
				# for detected double steps, we can simply interpolate to fill the gap
				if (x_diff == 2):
					old_x = thread_arr_seth_path[ID][-1][2]
					thread_arr_seth_path[ID].append(((clip_start_frame + idx), map_name, (seth_x + old_x)//2, seth_y, -1))
					move_cnt += 1
				if (y_diff == 2):
					old_y = thread_arr_seth_path[ID][-1][3]
					thread_arr_seth_path[ID].append(((clip_start_frame + idx), map_name, seth_x, (seth_y + old_y)//2, -1))
					move_cnt += 1

			# record the actual result of this frame
			thread_arr_seth_path[ID].append(((clip_start_frame + idx), map_name, seth_x, seth_y, max_val))
			move_cnt += 1
			
			# and update some stuff
			last_seth_x = seth_x
			last_seth_y = seth_y
			last_map_name = map_name
			current_map = get_map_by_name(map_name)
			ROI_is_updated = False
			break

	other_cnt = int(frame_cnt - (dupe_cnt + menu_cnt + move_cnt))
	# print(f"[THREAD #{ID}] Filtered out {dupe_cnt} ({100*dupe_cnt/frame_cnt:.2f}%) Dupe-Frames")
	# print(f"[THREAD #{ID}] Filtered out {menu_cnt} ({100*menu_cnt/frame_cnt:.2f}%) Menu-Frames.")
	# print(f"[THREAD #{ID}] Important Frames detected: {move_cnt} ({100*move_cnt/frame_cnt:.2f}%) Frames.")
	# print(f"[THREAD #{ID}] Unaccounted Frames: {other_cnt} ({100*other_cnt/frame_cnt:.2f}%) Frames.")
	thread_arr_dupe_cnt[ID] = dupe_cnt
	thread_arr_menu_cnt[ID] = menu_cnt
	thread_arr_move_cnt[ID] = move_cnt
	thread_arr_error_cnt[ID] = error_cnt
	thread_arr_other_cnt[ID] = other_cnt
	print(f"[THREAD #{ID}] -- Finished.")

thread_cnt = 4
target_start = 0 * 60 * 60
target_length = 0 * 60 * 60
running_start = target_start

# counter arrays
thread_arr_move_cnt = [0] * thread_cnt
thread_arr_dupe_cnt = [0] * thread_cnt
thread_arr_menu_cnt = [0] * thread_cnt
thread_arr_error_cnt = [0] * thread_cnt
thread_arr_other_cnt = [0] * thread_cnt
# more complex object arrays
threads = [None] * thread_cnt
trims = [None] * thread_cnt
# exhaustive list of seth locations
thread_arr_seth_path = [None] * thread_cnt



# initialize all the threads
for ID in range(0, thread_cnt):
	# reload the stream because appearently moviepy struggles otherwise...
	print(f"Preprocessing Stream Segment #{ID}...")
	stream = mp_editor.VideoFileClip("stream1.mp4")
	stream = stream.set_duration(stream.duration)
	if (target_length == 0):
		target_length = stream.duration
	# strip audio
	stream = stream.without_audio()
	# crop viewport
	stream = mp_video.fx.all.crop(stream, x1=168, x2=273, y1=45, y2=115)
	# lower FPS (instead of 60 FPS, only keep a couple of frames)
	stream = stream.set_fps(target_FPS)

	# trim the stream into the provided segment
	clip_start = running_start
	clip_end = clip_start + (target_length / thread_cnt) + thread_optimization(thread_cnt, ID, processing_factor=180)
	running_start = clip_end
	trims[ID] = stream.subclip(
		t_start = clip_start,
		t_end   = clip_end
	)
	# and get some metadata going (also its important to actually pass on the starting-frame for joining the data later)
	clip_start_frame = int(clip_start * target_FPS)
	frame_cnt = int(trims[ID].duration * target_FPS)
	print(f"-- Trimmed. {frame_cnt} Frames, Range = ({clip_start}-{clip_end})s, Duration = {trims[ID].duration}s")
	threads[ID] = threading.Thread(target=process_stream_segment, args=(ID, clip_start_frame))
	threads[ID].start()
	
# wait for them to finish, and write the results to a file while waiting
for ID in range(0, thread_cnt):
	threads[ID].join()
	


print("All Threads rejoined.")

# record the results
seth_path_file = open("seth_path.txt", "w")
seth_path_file.write(f"{'# Frame':<10}{'Map':<20}{'Seth X / Y':<15}{'Confidence'}\n")
for ID in range(0, thread_cnt):
	sepperator = f"# --- Thread #{ID} ---"
	seth_path_file.write(f"{sepperator:-<58}\n")
	for entry in thread_arr_seth_path[ID]:
		seth_path_file.write(f"{str(entry[0])+',':<10}{str(entry[1])+',':<20}{entry[2]:3d}, {entry[3]:3d},      {entry[4]:.3f}\n")

print("thread_arr_dupe_cnt:\t",  thread_arr_dupe_cnt,  "\tSUM = ", sum(thread_arr_dupe_cnt))
print("thread_arr_menu_cnt:\t",  thread_arr_menu_cnt,  "\tSUM = ", sum(thread_arr_menu_cnt))
print("thread_arr_move_cnt:\t",  thread_arr_move_cnt,  "\tSUM = ", sum(thread_arr_move_cnt))
print("thread_arr_error_cnt:\t", thread_arr_error_cnt, "\tSUM = ", sum(thread_arr_error_cnt))
# print("thread_arr_other_cnt: ", thread_arr_other_cnt, sum(thread_arr_other_cnt))
