import math
import real_measurements


def convert_normalized_to_pixel(normalized_coordinate, pixel_height):
    return normalized_coordinate * pixel_height

def calculate_distance(point1, point2):
    return math.sqrt((point2[0] - point1[0])**2 + (point2[1] - point1[1])**2)



data = real_measurements.landmark_data

# Normalized coordinates
shoulder1_normalized = data[11]#(0.4713554382324219, 0.3730059862136841)
shoulder2_normalized = data[12]#(0.37547117471694946, 0.37117284536361694)
hip1_normalized = data[23]#(0.4713554382324219, 0.3730059862136841)
hip2_normalized = data[24]#(0.37547117471694946, 0.37117284536361694)

# Real height and pixel height
real_height_cm =real_measurements.height #159.34002292963467
pixel_height = real_measurements.stable_height #557

# Convert normalized coordinates to pixel coordinates
shoulder1_pixel = (convert_normalized_to_pixel(shoulder1_normalized[0], pixel_height),
                convert_normalized_to_pixel(shoulder1_normalized[1], pixel_height))
shoulder2_pixel = (convert_normalized_to_pixel(shoulder2_normalized[0], pixel_height),
                convert_normalized_to_pixel(shoulder2_normalized[1], pixel_height))
hip1_pixel = (convert_normalized_to_pixel(hip1_normalized[0], pixel_height),
                convert_normalized_to_pixel(hip1_normalized[1], pixel_height))
hip2_pixel = (convert_normalized_to_pixel(hip2_normalized[0], pixel_height),
                convert_normalized_to_pixel(hip2_normalized[1], pixel_height))


# Calculate distance in pixels
shoulder_distance_pixels = calculate_distance(shoulder1_pixel , shoulder2_pixel)
hip_distance_pixels = calculate_distance(hip1_pixel , hip2_pixel)

# Convert distance to cm
shoulder_distance_inch = (real_height_cm / pixel_height) * shoulder_distance_pixels
hip_distance_inch = (real_height_cm / pixel_height) * hip_distance_pixels * 2 * 2

print(shoulder_distance_inch)
print(hip_distance_inch)


