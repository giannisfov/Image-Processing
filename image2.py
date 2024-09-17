import cv2
import matplotlib.pyplot as plt

# Φόρτωση εικόνας
bridge = cv2.imread('/content/gdrive/MyDrive/images-project-2/bridge.bmp', cv2.IMREAD_GRAYSCALE)
girlface = cv2.imread('/content/gdrive/MyDrive/images-project-2/girlface.bmp', cv2.IMREAD_GRAYSCALE)
lighthouse = cv2.imread('/content/gdrive/MyDrive/images-project-2/lighthouse.bmp', cv2.IMREAD_GRAYSCALE)

# Εμφάνιση εικόνας
plt.figure(figsize=(10, 4))
plt.subplot(1, 3, 1)
plt.imshow(bridge, cmap='gray')
plt.title('Bridge')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(girlface, cmap='gray')
plt.title('Girl Face')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(lighthouse, cmap='gray')
plt.title('Lighthouse')
plt.axis('off')

plt.show()


import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.filters import threshold_otsu

lighthouse = cv2.imread('/content/gdrive/MyDrive/images-project-2/lighthouse.bmp', cv2.IMREAD_GRAYSCALE)

sobelx = cv2.Sobel(lighthouse, cv2.CV_64F, 1, 0, ksize=3)
sobely = cv2.Sobel(lighthouse, cv2.CV_64F, 0, 1, ksize=3)
sobel_edges = np.hypot(sobelx, sobely)
sobel_edges = (sobel_edges / sobel_edges.max() * 255).astype(np.uint8)
otsu_threshold = threshold_otsu(sobel_edges)
sobel_binary = sobel_edges > otsu_threshold

roberts_cross_v = np.array([[1, 0], [0, -1]])
roberts_cross_h = np.array([[0, 1], [-1, 0]])
roberts_v = cv2.filter2D(lighthouse, -1, roberts_cross_v)
roberts_h = cv2.filter2D(lighthouse, -1, roberts_cross_h)
roberts_edges = np.hypot(roberts_v, roberts_h)
roberts_edges = (roberts_edges / roberts_edges.max() * 255).astype(np.uint8)
otsu_threshold = threshold_otsu(roberts_edges)
roberts_binary = roberts_edges > otsu_threshold


prewittx = cv2.filter2D(lighthouse, -1, cv2.getDerivKernels(1, 0, 3)[0])
prewitty = cv2.filter2D(lighthouse, -1, cv2.getDerivKernels(0, 1, 3)[0])
prewitt_edges = np.hypot(prewittx, prewitty)
prewitt_edges = (prewitt_edges / prewitt_edges.max() * 255).astype(np.uint8)
otsu_threshold = threshold_otsu(prewitt_edges)
prewitt_binary = prewitt_edges > otsu_threshold


kirsch_kernels = [np.array([[5, 5, 5], [-3, 0, -3], [-3, -3, -3]]), np.array([[-3, 5, 5], [-3, 0, 5], [-3, -3, -3]]),
                  np.array([[-3, -3, 5], [-3, 0, 5], [-3, -3, 5]]), np.array([[-3, -3, -3], [-3, 0, 5], [-3, 5, 5]]),
                  np.array([[-3, -3, -3], [-3, 0, -3], [5, 5, 5]]), np.array([[-3, -3, -3], [5, 0, -3], [5, 5, -3]]),
                  np.array([[5, -3, -3], [5, 0, -3], [5, -3, -3]]), np.array([[5, 5, -3], [5, 0, -3], [-3, -3, -3]])]

kirsch_edges = np.zeros_like(lighthouse, dtype=np.float64)
for kernel in kirsch_kernels:
    kirsch_response = cv2.filter2D(lighthouse, -1, kernel)
    kirsch_edges = np.maximum(kirsch_edges, kirsch_response)
kirsch_edges = (kirsch_edges / kirsch_edges.max() * 255).astype(np.uint8)
kirsch_threshold = 150  # Προσδιορίζεται πειραματικά
kirsch_binary = kirsch_edges > kirsch_threshold


plt.figure(figsize=(15, 10))

plt.subplot(2, 4, 1)
plt.imshow(lighthouse, cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(2, 4, 2)
plt.imshow(sobel_edges, cmap='gray')
plt.title('Sobel Edges')
plt.axis('off')

plt.subplot(2, 4, 3)
plt.imshow(roberts_edges, cmap='gray')
plt.title('Roberts Edges')
plt.axis('off')

plt.subplot(2, 4, 4)
plt.imshow(prewitt_edges, cmap='gray')
plt.title('Prewitt Edges')
plt.axis('off')

plt.subplot(2, 4, 5)
plt.imshow(sobel_binary, cmap='gray')
plt.title('Sobel Binary (Otsu)')
plt.axis('off')

plt.subplot(2, 4, 6)
plt.imshow(roberts_binary, cmap='gray')
plt.title('Roberts Binary (Otsu)')
plt.axis('off')

plt.subplot(2, 4, 7)
plt.imshow(prewitt_binary, cmap='gray')
plt.title('Prewitt Binary (Otsu)')
plt.axis('off')

plt.subplot(2, 4, 8)
plt.imshow(kirsch_binary, cmap='gray')
plt.title('Kirsch Binary')
plt.axis('off')

plt.show()

lighthouse = cv2.imread('/content/gdrive/MyDrive/images-project-2/lighthouse.bmp', cv2.IMREAD_GRAYSCALE)

# Εφαρμογή Gaussian Blur με διάφορες τιμές για τη διακύμανση
sigma_values = [1, 2, 3]
log_images = []

for sigma in sigma_values:
    blurred = cv2.GaussianBlur(lighthouse, (0, 0), sigma)
    log = cv2.Laplacian(blurred, cv2.CV_64F)
    log = (log / log.max() * 255).astype(np.uint8)
    log_images.append((sigma, log))

# Εύρεση του βέλτιστου κατωφλίου πειραματικά
threshold_values = [50, 100, 150, 200]
log_binary_images = []

for sigma, log_image in log_images:
    for threshold in threshold_values:
        _, log_binary = cv2.threshold(log_image, threshold, 255, cv2.THRESH_BINARY)
        log_binary_images.append((sigma, threshold, log_binary))

# Εμφάνιση των αποτελεσμάτων
fig, axes = plt.subplots(nrows=4, ncols=4, figsize=(15, 15))

axes[0, 0].imshow(lighthouse, cmap='gray')
axes[0, 0].set_title('Original Image')
axes[0, 0].axis('off')

i = 1
for sigma, log_image in log_images:
    row, col = divmod(i, 4)
    axes[row, col].imshow(log_image, cmap='gray')
    axes[row, col].set_title(f'LoG σ={sigma}')
    axes[row, col].axis('off')
    i += 1

for sigma, threshold, log_binary in log_binary_images:
    row, col = divmod(i, 4)
    axes[row, col].imshow(log_binary, cmap='gray')
    axes[row, col].set_title(f'σ={sigma}, t={threshold}')
    axes[row, col].axis('off')
    i += 1

plt.tight_layout()
plt.show()

# Υπολογισμός ακμών με τη μέθοδο Canny
canny_edges = cv2.Canny(lighthouse, 100, 200)  # Προεπιλεγμένες τιμές κατωφλίωσης

# Εμφάνιση της αρχικής εικόνας και των ακμών Canny
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(lighthouse, cmap='gray')
plt.title('Αρχική Εικόνα')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(canny_edges, cmap='gray')
plt.title('Ακμές Canny')
plt.axis('off')

plt.tight_layout()
plt.show()

import cv2
import numpy as np
import matplotlib.pyplot as plt

bridge = cv2.imread('/content/gdrive/MyDrive/images-project-2/bridge.bmp', cv2.IMREAD_GRAYSCALE)
girlface = cv2.imread('/content/gdrive/MyDrive/images-project-2/girlface.bmp', cv2.IMREAD_GRAYSCALE)
lighthouse = cv2.imread('/content/gdrive/MyDrive/images-project-2/lighthouse.bmp', cv2.IMREAD_GRAYSCALE)

def sobel_edge_detection(image):
    sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    sobel_edges = np.sqrt(sobelx**2 + sobely**2)
    sobel_edges = np.uint8(sobel_edges)
    return sobel_edges

def roberts_edge_detection(image):
    roberts_cross_v = np.array([[ 0, 0, 0 ],
                                 [ 0, 1, 0 ],
                                 [ 0, 0,-1 ]])
    roberts_cross_h = np.array([[ 0, 0, 0 ],
                                 [ 0, 1,-1 ],
                                 [ 0, 0, 0 ]])
    roberts_edges_v = cv2.filter2D(image, -1, roberts_cross_v)
    roberts_edges_h = cv2.filter2D(image, -1, roberts_cross_h)
    roberts_edges = np.sqrt(roberts_edges_v**2 + roberts_edges_h**2)
    roberts_edges = np.uint8(roberts_edges)
    return roberts_edges

def prewitt_edge_detection(image):
    kernelx = np.array([[ 1, 1, 1],
                        [ 0, 0, 0],
                        [-1,-1,-1]])
    kernely = np.array([[-1, 0, 1],
                        [-1, 0, 1],
                        [-1, 0, 1]])
    prewitt_edgesx = cv2.filter2D(image, -1, kernelx)
    prewitt_edgesy = cv2.filter2D(image, -1, kernely)
    prewitt_edges = np.sqrt(prewitt_edgesx**2 + prewitt_edgesy**2)
    prewitt_edges = np.uint8(prewitt_edges)
    return prewitt_edges

def kirsch_edge_detection(image):
    kernelG1 = np.array([[ 5, 5, 5],
                         [-3, 0,-3],
                         [-3,-3,-3]])

    kernelG2 = np.array([[-3, 5, 5],
                         [-3, 0, 5],
                         [-3,-3,-3]])

    kernelG3 = np.array([[-3,-3, 5],
                         [-3, 0, 5],
                         [-3,-3, 5]])

    kernelG4 = np.array([[-3,-3,-3],
                         [-3, 0, 5],
                         [-3, 5, 5]])

    kernelG5 = np.array([[-3,-3,-3],
                         [-3, 0,-3],
                         [ 5, 5, 5]])

    kernelG6 = np.array([[-3,-3,-3],
                         [ 5, 0,-3],
                         [ 5, 5,-3]])

    kernelG7 = np.array([[ 5,-3,-3],
                         [ 5, 0,-3],
                         [ 5,-3,-3]])

    kernelG8 = np.array([[ 5, 5,-3],
                         [ 5, 0,-3],
                         [-3,-3,-3]])

    kirsch_edges1 = cv2.filter2D(image, -1, kernelG1)
    kirsch_edges2 = cv2.filter2D(image, -1, kernelG2)
    kirsch_edges3 = cv2.filter2D(image, -1, kernelG3)
    kirsch_edges4 = cv2.filter2D(image, -1, kernelG4)
    kirsch_edges5 = cv2.filter2D(image, -1, kernelG5)
    kirsch_edges6 = cv2.filter2D(image, -1, kernelG6)
    kirsch_edges7 = cv2.filter2D(image, -1, kernelG7)
    kirsch_edges8 = cv2.filter2D(image, -1, kernelG8)

    kirsch_edges = np.maximum.reduce([kirsch_edges1, kirsch_edges2, kirsch_edges3,
                                      kirsch_edges4, kirsch_edges5, kirsch_edges6,
                                      kirsch_edges7, kirsch_edges8])
    return kirsch_edges

# Apply edge detection methods to images
bridge_sobel_edges = sobel_edge_detection(bridge)
bridge_roberts_edges = roberts_edge_detection(bridge)
bridge_prewitt_edges = prewitt_edge_detection(bridge)
bridge_kirsch_edges = kirsch_edge_detection(bridge)

girlface_sobel_edges = sobel_edge_detection(girlface)
girlface_roberts_edges = roberts_edge_detection(girlface)
girlface_prewitt_edges = prewitt_edge_detection(girlface)
girlface_kirsch_edges = kirsch_edge_detection(girlface)

# Plot the results
plt.figure(figsize=(15, 12))

import cv2
import numpy as np
import matplotlib.pyplot as plt

# Φόρτωση της εικόνας
image_paths = ["/content/gdrive/MyDrive/images-project-2/bridge.bmp", "/content/gdrive/MyDrive/images-project-2/girlface.bmp"]

# Καθορισμός μεθόδων και παραμέτρων
methods = ["Sobel", "Roberts", "Prewitt", "Kirsch", "LoG"]
sigma_values = [1, 2, 3]
threshold_values = [50, 100, 150, 200]
optimal_masks = {}

# Υπολογισμός ακμών για κάθε μέθοδο και εικόνα
for method in methods:
    if method != "LoG":
        optimal_masks[method] = {}
        for image_path in image_paths:
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            edges = []
            for sigma in sigma_values:
                blurred = cv2.GaussianBlur(image, (0, 0), sigma)
                if method == "Sobel":
                    edge = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
                elif method == "Roberts":
                    edge = cv2.Canny(blurred, 50, 100)
                elif method == "Prewitt":
                    edge = cv2.Sobel(blurred, cv2.CV_64F, 1, 1, ksize=3)
                elif method == "Kirsch":
                    edge = cv2.filter2D(blurred, cv2.CV_64F, np.array([[5, 5, 5], [-3, 0, -3], [-3, -3, -3]]))
                edge = np.uint8(np.absolute(edge))
                edges.append((sigma, edge))
            optimal_masks[method][image_path] = edges
    else:
        optimal_masks[method] = {}
        for image_path in image_paths:
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            edges = []
            for sigma in sigma_values:
                blurred = cv2.GaussianBlur(image, (0, 0), sigma)
                laplacian = cv2.Laplacian(blurred, cv2.CV_64F)
                laplacian = (laplacian / laplacian.max() * 255).astype(np.uint8)
                edges.append((sigma, laplacian))
            optimal_masks[method][image_path] = edges

# Εμφάνιση των βέλτιστων μασκών ακμών για κάθε μέθοδο και εικόνα
for method in methods:
    for image_path in image_paths:
        plt.figure(figsize=(15, 5))
        plt.suptitle(f"{method} Edges ", fontsize=16)
        for i, (sigma, edge) in enumerate(optimal_masks[method][image_path], start=1):
            plt.subplot(1, len(sigma_values), i)
            plt.imshow(edge, cmap='gray')
            plt.title(f"σ={sigma}")
            plt.axis('off')
        plt.show()

# Συνάρτηση για τη δημιουργία του δέντρου Huffman
def build_huffman_tree(freq_map):
    heap = [[weight, [symbol, '']] for symbol, weight in freq_map.items()]
    heapify(heap)
    while len(heap) > 1:
        lo = heappop(heap)
        hi = heappop(heap)
        for pair in lo[1:]:
            pair[1] = '0' + pair[1]
        for pair in hi[1:]:
            pair[1] = '1' + pair[1]
        heappush(heap, [lo[0] + hi[0]] + lo[1:] + hi[1:])
    return heap[0][1:]

# Συνάρτηση για την κωδικοποίηση της εικόνας με το δέντρο Huffman
def encode_image(image, huffman_tree):
    encoded_image = ''
    for pixel in image.flatten():
        encoded_image += huffman_tree[pixel]
    return encoded_image

# Φόρτωση της εικόνας
def load_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    return image

# Υπολογισμός συχνοτήτων εμφάνισης επιπέδων γκρι
def calculate_frequencies(image):
    freq_map = defaultdict(int)
    for pixel in image.flatten():
        freq_map[pixel] += 1
    return freq_map

# Κωδικοποίηση και αποθήκευση της κωδικοποιημένης εικόνας σε αρχείο
def save_encoded_image(encoded_image, output_path):
    with open(output_path, 'w') as f:
        f.write(encoded_image)

# Φόρτωση της εικόνας
image_paths = ["/content/gdrive/MyDrive/images-project-2/bridge.bmp", "/content/gdrive/MyDrive/images-project-2/girlface.bmp", "/content/gdrive/MyDrive/images-project-2/lighthouse.bmp"]

# Κωδικοποίηση και αποθήκευση των κωδικοποιημένων εικόνων
for image_path in image_paths:
    image = load_image(image_path)
    freq_map = calculate_frequencies(image)
    huffman_tree = dict(build_huffman_tree(freq_map))
    encoded_image = encode_image(image, huffman_tree)
    output_path = f"{image_path.split('.')[0]}_encoded.txt"
    save_encoded_image(encoded_image, output_path)
    print(f"Η εικόνα {image_path} κωδικοποιήθηκε με επιτυχία και αποθηκεύτηκε στο αρχείο {output_path}.")

import cv2
import numpy as np
from heapq import heappush, heappop, heapify
from collections import defaultdict
import os

def calculate_metrics(image, encoded_size, freq_map):
    # Υπολογισμός μέσου μήκους κωδικής λέξης
    total_symbols = sum(freq_map.values())
    average_code_length = encoded_size / total_symbols

    # Υπολογισμός εντροπίας
    entropy = 0
    for freq in freq_map.values():
        probability = freq / total_symbols
        entropy -= probability * np.log2(probability)

    # Υπολογισμός λόγου συμπίεσης
    original_size = image.size * 8
    compression_ratio = original_size / encoded_size

    return average_code_length, entropy, compression_ratio

# Φόρτωση της εικόνας
image_paths = ["/content/gdrive/MyDrive/images-project-2/bridge.bmp", "/content/gdrive/MyDrive/images-project-2/girlface.bmp", "/content/gdrive/MyDrive/images-project-2/lighthouse.bmp"]
for image_path in image_paths:
    image = load_image(image_path)

    # Υπολογισμός συχνοτήτων εμφάνισης επιπέδων γκρι
    freq_map = calculate_frequencies(image)

    # Κωδικοποίηση της εικόνας
    huffman_tree = dict(build_huffman_tree(freq_map))
    encoded_image = encode_image(image, huffman_tree)

    # Αποθήκευση της κωδικοποιημένης εικόνας σε αρχείο
    output_path = f"{image_path.split('.')[0]}_encoded.txt"
    save_encoded_image(encoded_image, output_path)

    # Υπολογισμός των μετρικών
    encoded_size = os.path.getsize(output_path) * 8  # Μέγεθος σε bits
    average_code_length, entropy, compression_ratio = calculate_metrics(image, encoded_size, freq_map)

    # Εκτύπωση των μετρικών
    print(f"Για την εικόνα {image_path}:")
    print(f"Κωδικές λέξεις: {huffman_tree}")
    print(f"Μέσο μήκος κωδικής λέξης: {average_code_length}")
    print(f"Εντροπία: {entropy}")
    print(f"Λόγος συμπίεσης: {compression_ratio}")

Q50 = np.array([
    [16, 11, 10, 16, 24, 40, 51, 61],
    [12, 12, 14, 19, 26, 58, 60, 55],
    [14, 13, 16, 24, 40, 57, 69, 56],
    [14, 17, 22, 29, 51, 87, 80, 62],
    [18, 22, 37, 56, 68, 109, 103, 77],
    [24, 35, 55, 64, 81, 104, 113, 92],
    [49, 64, 78, 87, 103, 121, 120, 101],
    [72, 92, 95, 98, 112, 100, 103, 99]
])

def scale_quantization_matrix(Q, quality_factor):
    if quality_factor < 50:
        scale = 5000 / quality_factor
    else:
        scale = 200 - 2 * quality_factor
    Q_scaled = np.floor((Q * scale + 50) / 100)
    Q_scaled[Q_scaled == 0] = 1
    return Q_scaled

Q10 = scale_quantization_matrix(Q50, 10)
Q50 = scale_quantization_matrix(Q50, 50)

import cv2
import numpy as np
import matplotlib.pyplot as plt
import scipy.fftpack

# DCT και αντίστροφη DCT
def dct2(block):
    return scipy.fftpack.dct(scipy.fftpack.dct(block.T, norm='ortho').T, norm='ortho')

def idct2(block):
    return scipy.fftpack.idct(scipy.fftpack.idct(block.T, norm='ortho').T, norm='ortho')

# Κβάντιση και αποκβάντιση
def quantize(block, quant_matrix):
    return np.round(block / quant_matrix)

def dequantize(block, quant_matrix):
    return block * quant_matrix

# Συμπίεση JPEG
def jpeg_compression(image, quant_matrix):
    height, width = image.shape
    compressed = np.zeros_like(image, dtype=np.float32)

    for i in range(0, height, 8):
        for j in range(0, width, 8):
            block = image[i:i+8, j:j+8]
            dct_block = dct2(block)
            quant_block = quantize(dct_block, quant_matrix)
            compressed[i:i+8, j:j+8] = quant_block

    return compressed

# Αποσυμπίεση JPEG
def jpeg_decompression(compressed, quant_matrix):
    height, width = compressed.shape
    decompressed = np.zeros_like(compressed, dtype=np.float32)

    for i in range(0, height, 8):
        for j in range(0, width, 8):
            quant_block = compressed[i:i+8, j:j+8]
            dequant_block = dequantize(quant_block, quant_matrix)
            block = idct2(dequant_block)
            decompressed[i:i+8, j:j+8] = block

    return decompressed

# Φόρτωση εικόνας
image = cv2.imread("/content/gdrive/MyDrive/images-project-2/lighthouse.bmp", cv2.IMREAD_GRAYSCALE)
image2 = cv2.imread("/content/gdrive/MyDrive/images-project-2/bridge.bmp", cv2.IMREAD_GRAYSCALE)
image3 = cv2.imread("/content/gdrive/MyDrive/images-project-2/girlface.bmp", cv2.IMREAD_GRAYSCALE)

# Συμπίεση και αποκωδικοποίηση εικόνας με Q10
compressed_Q10 = jpeg_compression(image, Q10)
decompressed_Q10 = jpeg_decompression(compressed_Q10, Q10)

# Συμπίεση και αποκωδικοποίηση εικόνας με Q50
compressed_Q50 = jpeg_compression(image, Q50)
decompressed_Q50 = jpeg_decompression(compressed_Q50, Q50)

# Παρουσίαση αποτελεσμάτων
plt.figure(figsize=(12, 8))

plt.subplot(2, 3, 1)
plt.imshow(image, cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(2, 3, 2)
plt.imshow(compressed_Q10, cmap='gray')
plt.title('Compressed Q10')
plt.axis('off')

plt.subplot(2, 3, 3)
plt.imshow(decompressed_Q10, cmap='gray')
plt.title('Decompressed Q10')
plt.axis('off')

plt.subplot(2, 3, 5)
plt.imshow(compressed_Q50, cmap='gray')
plt.title('Compressed Q50')
plt.axis('off')

plt.subplot(2, 3, 6)
plt.imshow(decompressed_Q50, cmap='gray')
plt.title('Decompressed Q50')
plt.axis('off')

plt.show()

import cv2
import numpy as np
import matplotlib.pyplot as plt
import scipy.fftpack
from collections import Counter
from heapq import heappush, heappop, heapify

# DCT και αντίστροφη DCT
def dct2(block):
    return scipy.fftpack.dct(scipy.fftpack.dct(block.T, norm='ortho').T, norm='ortho')

def idct2(block):
    return scipy.fftpack.idct(scipy.fftpack.idct(block.T, norm='ortho').T, norm='ortho')

# Κβάντιση και αποκβάντιση
def quantize(block, quant_matrix):
    return np.round(block / quant_matrix)

def dequantize(block, quant_matrix):
    return block * quant_matrix

# Huffman κωδικοποίηση
class Node:
    def __init__(self, frequency, symbol, left=None, right=None):
        self.frequency = frequency
        self.symbol = symbol
        self.left = left
        self.right = right
        self.huff = ''

def calculate_codes(node, val=''):
    codes = {}
    new_val = val + str(node.huff)
    if node.left:
        codes.update(calculate_codes(node.left, new_val))
    if node.right:
        codes.update(calculate_codes(node.right, new_val))
    if not node.left and not node.right:
        codes[node.symbol] = new_val
    return codes

def huffman_encoding(data):
    symbol_freq = Counter(data)
    heap = [Node(freq, symbol) for symbol, freq in symbol_freq.items()]
    heapify(heap)

    while len(heap) > 1:
        left = heappop(heap)
        right = heappop(heap)
        left.huff = 0
        right.huff = 1
        new_node = Node(left.frequency + right.frequency, left.symbol + right.symbol, left, right)
        heappush(heap, new_node)

    huffman_tree = heappop(heap)
    huffman_codes = calculate_codes(huffman_tree)

    encoded_data = ''.join([huffman_codes[symbol] for symbol in data])
    return encoded_data, huffman_codes

# Υπολογισμός του μέσου μήκους κωδικής λέξης
def calculate_average_code_length(huffman_codes, symbol_freq):
    total_length = sum(len(huffman_codes[symbol]) * freq for symbol, freq in symbol_freq.items())
    total_symbols = sum(symbol_freq.values())
    return total_length / total_symbols

# Συμπίεση JPEG με Huffman κωδικοποίηση
def jpeg_compression_with_huffman(image, quant_matrix):
    height, width = image.shape
    compressed = np.zeros_like(image, dtype=np.float32)
    all_quant_blocks = []

    for i in range(0, height, 8):
        for j in range(0, width, 8):
            block = image[i:i+8, j:j+8]
            dct_block = dct2(block)
            quant_block = quantize(dct_block, quant_matrix)
            compressed[i:i+8, j:j+8] = quant_block
            all_quant_blocks.extend(quant_block.flatten())

    # Huffman κωδικοποίηση
    encoded_data, huffman_codes = huffman_encoding(all_quant_blocks)
    symbol_freq = Counter(all_quant_blocks)
    avg_code_length = calculate_average_code_length(huffman_codes, symbol_freq)

    return compressed, encoded_data, avg_code_length

# Αποσυμπίεση JPEG
def jpeg_decompression_with_huffman(compressed, quant_matrix):
    height, width = compressed.shape
    decompressed = np.zeros_like(compressed, dtype=np.float32)

    for i in range(0, height, 8):
        for j in range(0, width, 8):
            quant_block = compressed[i:i+8, j:j+8]
            dequant_block = dequantize(quant_block, quant_matrix)
            block = idct2(dequant_block)
            decompressed[i:i+8, j:j+8] = block

    return decompressed

# Παράμετροι κβάντισης
Q50 = np.array([
    [16, 11, 10, 16, 24, 40, 51, 61],
    [12, 12, 14, 19, 26, 58, 60, 55],
    [14, 13, 16, 24, 40, 57, 69, 56],
    [14, 17, 22, 29, 51, 87, 80, 62],
    [18, 22, 37, 56, 68, 109, 103, 77],
    [24, 35, 55, 64, 81, 104, 113, 92],
    [49, 64, 78, 87, 103, 121, 120, 101],
    [72, 92, 95, 98, 112, 100, 103, 99]
])

Q10 = scale_quantization_matrix(Q50, 10)

# Φόρτωση εικόνων
#image_names = ["/content/gdrive/MyDrive/images-project-2/bridge.bmp", 'girlface.bmp', 'lighthouse.bmp']
image_names = ["/content/gdrive/MyDrive/images-project-2/bridge.bmp", "/content/gdrive/MyDrive/images-project-2/girlface.bmp", "/content/gdrive/MyDrive/images-project-2/lighthouse.bmp"]
quant_matrices = {'Q10': Q10, 'Q50': Q50}

# Συμπίεση και αποκωδικοποίηση εικόνων
results = {}
for image_name in image_names:
    image = cv2.imread(image_name, cv2.IMREAD_GRAYSCALE)
    results[image_name] = {}
    for qname, quant_matrix in quant_matrices.items():
        compressed, encoded_data, avg_code_length = jpeg_compression_with_huffman(image, quant_matrix)
        decompressed = jpeg_decompression_with_huffman(compressed, quant_matrix)

        original_size = image.size * 8  # σε bits
        compressed_size = len(encoded_data)
        compression_ratio = original_size / compressed_size

        results[image_name][qname] = {
            'avg_code_length': avg_code_length,
            'compression_ratio': compression_ratio,
            'decompressed_image': decompressed
        }

# Παρουσίαση αποτελεσμάτων
for image_name, res in results.items():
    print(f"Results for {image_name}:")
    for qname, metrics in res.items():
        print(f"  Quantization: {qname}")
        print(f"    Average Code Length: {metrics['avg_code_length']:.2f} bits")
        print(f"    Compression Ratio: {metrics['compression_ratio']:.2f}")
    print()

    plt.figure(figsize=(12, 8))
    plt.subplot(1, 3, 1)
    plt.imshow(cv2.imread(image_name, cv2.IMREAD_GRAYSCALE), cmap='gray')
    plt.title(f'Original {image_name}')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(res['Q10']['decompressed_image'], cmap='gray')
    plt.title(f'Decompressed Q10')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(res['Q50']['decompressed_image'], cmap='gray')
    plt.title(f'Decompressed Q50')
    plt.axis('off')

    plt.show()
import cv2
import numpy as np
import matplotlib.pyplot as plt
import scipy.fftpack
from collections import Counter
from heapq import heappush, heappop, heapify

# DCT και αντίστροφη DCT
def dct2(block):
    return scipy.fftpack.dct(scipy.fftpack.dct(block.T, norm='ortho').T, norm='ortho')

def idct2(block):
    return scipy.fftpack.idct(scipy.fftpack.idct(block.T, norm='ortho').T, norm='ortho')

# Κβάντιση και αποκβάντιση
def quantize(block, quant_matrix):
    return np.round(block / quant_matrix)

def dequantize(block, quant_matrix):
    return block * quant_matrix

# Huffman κωδικοποίηση
class Node:
    def __init__(self, frequency, symbol, left=None, right=None):
        self.frequency = frequency
        self.symbol = symbol
        self.left = left
        self.right = right
        self.huff = ''

    # Σύγκριση κόμβων με βάση τη συχνότητα
    def __lt__(self, other):
        return self.frequency < other.frequency

def calculate_codes(node, val=''):
    codes = {}
    new_val = val + str(node.huff)
    if node.left:
        codes.update(calculate_codes(node.left, new_val))
    if node.right:
        codes.update(calculate_codes(node.right, new_val))
    if not node.left and not node.right:
        codes[node.symbol] = new_val
    return codes

def huffman_encoding(data):
    symbol_freq = Counter(data)
    heap = [Node(freq, symbol) for symbol, freq in symbol_freq.items()]
    heapify(heap)

    while len(heap) > 1:
        left = heappop(heap)
        right = heappop(heap)
        left.huff = '0'
        right.huff = '1'
        new_node = Node(left.frequency + right.frequency, left.symbol + right.symbol, left, right)
        heappush(heap, new_node)

    huffman_tree = heappop(heap)
    huffman_codes = calculate_codes(huffman_tree)

    encoded_data = ''.join([huffman_codes[symbol] for symbol in data])
    return encoded_data, huffman_codes

# Υπολογισμός του μέσου μήκους κωδικής λέξης
def calculate_average_code_length(huffman_codes, symbol_freq):
    total_length = sum(len(huffman_codes[symbol]) * freq for symbol, freq in symbol_freq.items())
    total_symbols = sum(symbol_freq.values())
    return total_length / total_symbols

# Συμπίεση JPEG με Huffman κωδικοποίηση
def jpeg_compression_with_huffman(image, quant_matrix):
    height, width = image.shape
    compressed = np.zeros_like(image, dtype=np.float32)
    all_quant_blocks = []

    for i in range(0, height, 8):
        for j in range(0, width, 8):
            block = image[i:i+8, j:j+8]
            dct_block = dct2(block)
            quant_block = quantize(dct_block, quant_matrix)
            compressed[i:i+8, j:j+8] = quant_block
            all_quant_blocks.extend(quant_block.flatten())

    # Huffman κωδικοποίηση
    encoded_data, huffman_codes = huffman_encoding(all_quant_blocks)
    symbol_freq = Counter(all_quant_blocks)
    avg_code_length = calculate_average_code_length(huffman_codes, symbol_freq)

    return compressed, encoded_data, avg_code_length

# Αποσυμπίεση JPEG
def jpeg_decompression_with_huffman(compressed, quant_matrix):
    height, width = compressed.shape
    decompressed = np.zeros_like(compressed, dtype=np.float32)

    for i in range(0, height, 8):
        for j in range(0, width, 8):
            quant_block = compressed[i:i+8, j:j+8]
            dequant_block = dequantize(quant_block, quant_matrix)
            block = idct2(dequant_block)
            decompressed[i:i+8, j:j+8] = block

    return decompressed

# Συνάρτηση για κλιμάκωση πίνακα κβάντισης
def scale_quantization_matrix(Q, scale):
    scale_factor = 50 / scale if scale < 50 else 2 - (scale / 50)
    return np.floor((Q * scale_factor + 50) / 100)

# Παράμετροι κβάντισης
Q50 = np.array([
    [16, 11, 10, 16, 24, 40, 51, 61],
    [12, 12, 14, 19, 26, 58, 60, 55],
    [14, 13, 16, 24, 40, 57, 69, 56],
    [14, 17, 22, 29, 51, 87, 80, 62],
    [18, 22, 37, 56, 68, 109, 103, 77],
    [24, 35, 55, 64, 81, 104, 113, 92],
    [49, 64, 78, 87, 103, 121, 120, 101],
    [72, 92, 95, 98, 112, 100, 103, 99]
])

Q10 = scale_quantization_matrix(Q50, 10)

# Φόρτωση εικόνων από το Google Drive
image_names = ['/content/gdrive/MyDrive/images-project-2/bridge.bmp',
               '/content/gdrive/MyDrive/images-project-2/girlface.bmp',
               '/content/gdrive/MyDrive/images-project-2/lighthouse.bmp']
quant_matrices = {'Q10': Q10, 'Q50': Q50}

# Συμπίεση και αποκωδικοποίηση εικόνων
results = {}
for image_name in image_names:
    image = cv2.imread(image_name, cv2.IMREAD_GRAYSCALE)
    results[image_name] = {}
    for qname, quant_matrix in quant_matrices.items():
        compressed, encoded_data, avg_code_length = jpeg_compression_with_huffman(image, quant_matrix)
        decompressed = jpeg_decompression_with_huffman(compressed, quant_matrix)

        original_size = image.size * 8  # σε bits
        compressed_size = len(encoded_data)
        compression_ratio = original_size / compressed_size

        results[image_name][qname] = {
            'avg_code_length': avg_code_length,
            'compression_ratio': compression_ratio,
            'decompressed_image': decompressed
        }

# Παρουσίαση αποτελεσμάτων
for image_name, res in results.items():
    print(f"Results for {image_name}:")
    for qname, metrics in res.items():
        print(f"  Quantization: {qname}")
        print(f"    Average Code Length: {metrics['avg_code_length']:.2f} bits")
        print(f"    Compression Ratio: {metrics['compression_ratio']:.2f}")
    print()

    plt.figure(figsize=(12, 8))
    plt.subplot(1, 3, 1)
    plt.imshow(cv2.imread(image_name, cv2.IMREAD_GRAYSCALE), cmap='gray')
    plt.title(f'Original {image_name.split("/")[-1]}')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(res['Q10']['decompressed_image'], cmap='gray')
    plt.title(f'Decompressed Q10')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(res['Q50']['decompressed_image'], cmap='gray')
    plt.title(f'Decompressed Q50')
    plt.axis('off')

    plt.show()

import cv2
import numpy as np
import matplotlib.pyplot as plt
import scipy.fftpack
from collections import Counter
from heapq import heappush, heappop, heapify

# DCT και αντίστροφη DCT
def dct2(block):
    return scipy.fftpack.dct(scipy.fftpack.dct(block.T, norm='ortho').T, norm='ortho')

def idct2(block):
    return scipy.fftpack.idct(scipy.fftpack.idct(block.T, norm='ortho').T, norm='ortho')

# Κβάντιση και αποκβάντιση
def quantize(block, quant_matrix):
    return np.round(block / quant_matrix)

def dequantize(block, quant_matrix):
    return block * quant_matrix

# Huffman κωδικοποίηση
class Node:
    def __init__(self, frequency, symbol, left=None, right=None):
        self.frequency = frequency
        self.symbol = symbol
        self.left = left
        self.right = right
        self.huff = ''

    # Σύγκριση κόμβων με βάση τη συχνότητα
    def __lt__(self, other):
        return self.frequency < other.frequency

def calculate_codes(node, val=''):
    codes = {}
    new_val = val + str(node.huff)
    if node.left:
        codes.update(calculate_codes(node.left, new_val))
    if node.right:
        codes.update(calculate_codes(node.right, new_val))
    if not node.left and not node.right:
        codes[node.symbol] = new_val
    return codes

def huffman_encoding(data):
    symbol_freq = Counter(data)
    heap = [Node(freq, symbol) for symbol, freq in symbol_freq.items()]
    heapify(heap)

    while len(heap) > 1:
        left = heappop(heap)
        right = heappop(heap)
        left.huff = '0'
        right.huff = '1'
        new_node = Node(left.frequency + right.frequency, left.symbol + right.symbol, left, right)
        heappush(heap, new_node)

    huffman_tree = heappop(heap)
    huffman_codes = calculate_codes(huffman_tree)

    encoded_data = ''.join([huffman_codes[symbol] for symbol in data])
    return encoded_data, huffman_codes

# Υπολογισμός του μέσου μήκους κωδικής λέξης
def calculate_average_code_length(huffman_codes, symbol_freq):
    total_length = sum(len(huffman_codes[symbol]) * freq for symbol, freq in symbol_freq.items())
    total_symbols = sum(symbol_freq.values())
    return total_length / total_symbols

# Συμπίεση JPEG με Huffman κωδικοποίηση
def jpeg_compression_with_huffman(image, quant_matrix):
    height, width = image.shape
    compressed = np.zeros_like(image, dtype=np.float32)
    all_quant_blocks = []

    for i in range(0, height, 8):
        for j in range(0, width, 8):
            block = image[i:i+8, j:j+8]
            dct_block = dct2(block)
            quant_block = quantize(dct_block, quant_matrix)
            compressed[i:i+8, j:j+8] = quant_block
            all_quant_blocks.extend(quant_block.flatten())

    # Huffman κωδικοποίηση
    encoded_data, huffman_codes = huffman_encoding(all_quant_blocks)
    symbol_freq = Counter(all_quant_blocks)
    avg_code_length = calculate_average_code_length(huffman_codes, symbol_freq)

    return compressed, encoded_data, avg_code_length

# Αποσυμπίεση JPEG
def jpeg_decompression_with_huffman(compressed, quant_matrix):
    height, width = compressed.shape
    decompressed = np.zeros_like(compressed, dtype=np.float32)

    for i in range(0, height, 8):
        for j in range(0, width, 8):
            quant_block = compressed[i:i+8, j:j+8]
            dequant_block = dequantize(quant_block, quant_matrix)
            block = idct2(dequant_block)
            decompressed[i:i+8, j:j+8] = block

    return decompressed

# Συνάρτηση για κλιμάκωση πίνακα κβάντισης
def scale_quantization_matrix(Q, scale):
    scale_factor = 50 / scale if scale < 50 else 2 - (scale / 50)
    return np.floor((Q * scale_factor + 50) / 100)

# Υπολογισμός PSNR
def calculate_psnr(original, decompressed):
    mse = np.mean((original - decompressed) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr

# Παράμετροι κβάντισης
Q50 = np.array([
    [16, 11, 10, 16, 24, 40, 51, 61],
    [12, 12, 14, 19, 26, 58, 60, 55],
    [14, 13, 16, 24, 40, 57, 69, 56],
    [14, 17, 22, 29, 51, 87, 80, 62],
    [18, 22, 37, 56, 68, 109, 103, 77],
    [24, 35, 55, 64, 81, 104, 113, 92],
    [49, 64, 78, 87, 103, 121, 120, 101],
    [72, 92, 95, 98, 112, 100, 103, 99]
])

Q10 = scale_quantization_matrix(Q50, 10)

# Φόρτωση εικόνων από το Google Drive
image_names = ['/content/gdrive/MyDrive/images-project-2/bridge.bmp',
               '/content/gdrive/MyDrive/images-project-2/girlface.bmp',
               '/content/gdrive/MyDrive/images-project-2/lighthouse.bmp']
quant_matrices = {'Q10': Q10, 'Q50': Q50}

# Συμπίεση και αποκωδικοποίηση εικόνων
results = {}
for image_name in image_names:
    image = cv2.imread(image_name, cv2.IMREAD_GRAYSCALE)
    results[image_name] = {}
    for qname, quant_matrix in quant_matrices.items():
        compressed, encoded_data, avg_code_length = jpeg_compression_with_huffman(image, quant_matrix)
        decompressed = jpeg_decompression_with_huffman(compressed, quant_matrix)

        original_size = image.size * 8  # σε bits
        compressed_size = len(encoded_data)
        compression_ratio = original_size / compressed_size

        psnr = calculate_psnr(image, decompressed)

        results[image_name][qname] = {
            'avg_code_length': avg_code_length,
            'compression_ratio': compression_ratio,
            'psnr': psnr,
            'decompressed_image': decompressed
        }

# Παρουσίαση αποτελεσμάτων
for image_name, res in results.items():
    print(f"Results for {image_name}:")
    for qname, metrics in res.items():
        print(f"  Quantization: {qname}")
        print(f"    Average Code Length: {metrics['avg_code_length']:.2f} bits")
        print(f"    Compression Ratio: {metrics['compression_ratio']:.2f}")
        print(f"    PSNR: {metrics['psnr']:.2f} dB")
    print()

    plt.figure(figsize=(12, 8))
    plt.subplot(1, 3, 1)
    plt.imshow(cv2.imread(image_name, cv2.IMREAD_GRAYSCALE), cmap='gray')
    plt.title(f'Original {image_name.split("/")[-1]}')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(res['Q10']['decompressed_image'], cmap='gray')
    plt.title(f'Decompressed Q10\nPSNR: {res["Q10"]["psnr"]:.2f} dB')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(res['Q50'])

import cv2
import numpy as np
import matplotlib.pyplot as plt
import scipy.fftpack
from collections import Counter
from heapq import heappush, heappop, heapify

# DCT και αντίστροφη DCT
def dct2(block):
    return scipy.fftpack.dct(scipy.fftpack.dct(block.T, norm='ortho').T, norm='ortho')

def idct2(block):
    return scipy.fftpack.idct(scipy.fftpack.idct(block.T, norm='ortho').T, norm='ortho')


def quantize(block, quant_matrix):
    return np.round(block / quant_matrix)

def dequantize(block, quant_matrix):
    return block * quant_matrix

# Huffman κωδικοποίηση
class Node:
    def __init__(self, frequency, symbol, left=None, right=None):
        self.frequency = frequency
        self.symbol = symbol
        self.left = left
        self.right = right
        self.huff = ''

    def __lt__(self, other):
        return self.frequency < other.frequency

def calculate_codes(node, val=''):
    codes = {}
    new_val = val + str(node.huff)
    if node.left:
        codes.update(calculate_codes(node.left, new_val))
    if node.right:
        codes.update(calculate_codes(node.right, new_val))
    if not node.left and not node.right:
        codes[node.symbol] = new_val
    return codes

def huffman_encoding(data):
    symbol_freq = Counter(data)
    heap = [Node(freq, symbol) for symbol, freq in symbol_freq.items()]
    heapify(heap)

    while len(heap) > 1:
        left = heappop(heap)
        right = heappop(heap)
        left.huff = '0'
        right.huff = '1'
        new_node = Node(left.frequency + right.frequency, left.symbol + left.symbol + right.symbol, left, right)
        heappush(heap, new_node)

    huffman_tree = heappop(heap)
    huffman_codes = calculate_codes(huffman_tree)

    encoded_data = ''.join([huffman_codes[symbol] for symbol in data])
    return encoded_data, huffman_codes


def calculate_average_code_length(huffman_codes, symbol_freq):
    total_length = sum(len(huffman_codes[symbol]) * freq for symbol, freq in symbol_freq.items())
    total_symbols = sum(symbol_freq.values())
    return total_length / total_symbols


def jpeg_compression_with_huffman(image, quant_matrix):
    height, width = image.shape
    compressed = np.zeros_like(image, dtype=np.float32)
    all_quant_blocks = []

    for i in range(0, height, 8):
        for j in range(0, width, 8):
            block = image[i:i+8, j:j+8]
            dct_block = dct2(block)
            quant_block = quantize(dct_block, quant_matrix)
            compressed[i:i+8, j:j+8] = quant_block
            all_quant_blocks.extend(quant_block.flatten())


    encoded_data, huffman_codes = huffman_encoding(all_quant_blocks)
    symbol_freq = Counter(all_quant_blocks)
    avg_code_length = calculate_average_code_length(huffman_codes, symbol_freq)

    return compressed, encoded_data, avg_code_length

# Αποσυμπίεση JPEG
def jpeg_decompression_with_huffman(compressed, quant_matrix):
    height, width = compressed.shape
    decompressed = np.zeros_like(compressed, dtype=np.float32)

    for i in range(0, height, 8):
        for j in range(0, width, 8):
            quant_block = compressed[i:i+8, j:j+8]
            dequant_block = dequantize(quant_block, quant_matrix)
            block = idct2(dequant_block)
            decompressed[i:i+8, j:j+8] = block

    return decompressed


def scale_quantization_matrix(Q, scale):
    scale_factor = 50 / scale if scale < 50 else 2 - (scale / 50)
    return np.floor((Q * scale_factor + 50) / 100)

# Υπολογισμός PSNR
def calculate_psnr(original, decompressed):
    mse = np.mean((original - decompressed) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr


Q50 = np.array([
    [16, 11, 10, 16, 24, 40, 51, 61],
    [12, 12, 14, 19, 26, 58, 60, 55],
    [14, 13, 16, 24, 40, 57, 69, 56],
    [14, 17, 22, 29, 51, 87, 80, 62],
    [18, 22, 37, 56, 68, 109, 103, 77],
    [24, 35, 55, 64, 81, 104, 113, 92],
    [49, 64, 78, 87, 103, 121, 120, 101],
    [72, 92, 95, 98, 112, 100, 103, 99]
])

Q10 = scale_quantization_matrix(Q50, 10)


image_names = ['/content/gdrive/MyDrive/images-project-2/bridge.bmp',
               '/content/gdrive/MyDrive/images-project-2/girlface.bmp',
               '/content/gdrive/MyDrive/images-project-2/lighthouse.bmp']
quant_matrices = {'Q10': Q10, 'Q50': Q50}


results = {}
for image_name in image_names:
    image = cv2.imread(image_name, cv2.IMREAD_GRAYSCALE)
    results[image_name] = {}
    for qname, quant_matrix in quant_matrices.items():
        compressed, encoded_data, avg_code_length = jpeg_compression_with_huffman(image, quant_matrix)
        decompressed = jpeg_decompression_with_huffman(compressed, quant_matrix)

        original_size = image.size * 8  # σε bits
        compressed_size = len(encoded_data)
        compression_ratio = original_size / compressed_size

        psnr = calculate_psnr(image, decompressed)

        results[image_name][qname] = {
            'avg_code_length': avg_code_length,
            'compression_ratio': compression_ratio,
            'psnr': psnr,
            'decompressed_image': decompressed
        }


for image_name, res in results.items():
    print(f"Results for {image_name}:")
    for qname, metrics in res.items():
        print(f"  Quantization: {qname}")
        print(f"    Average Code Length: {metrics['avg_code_length']:.2f} bits")
        print(f"    Compression Ratio: {metrics['compression_ratio']:.2f}")
        print(f"    PSNR: {metrics['psnr']:.2f} dB")
    print()

    plt.figure(figsize=(12, 8))
    plt.subplot(1, 3, 1)
    plt.imshow(cv2.imread(image_name, cv2.IMREAD_GRAYSCALE), cmap='gray')
    plt.title(f'Original {image_name.split("/")[-1]}')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(res['Q10']['decompressed_image'], cmap='gray')
    plt.title(f'Decompressed Q10\nPSNR: {res["Q10"]["psnr"]:.2f} dB')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(res['Q50']['decompressed_image'], cmap='gray')
    plt.title(f'Decompressed Q50\nPSNR: {res["Q50"]["psnr"]:.2f} dB')
    plt.axis('off')

    plt.show()
import cv2
import numpy as np
import matplotlib.pyplot as plt
import scipy.fftpack
from collections import Counter
from heapq import heappush, heappop, heapify

# DCT και αντίστροφη DCT
def dct2(block):
    return scipy.fftpack.dct(scipy.fftpack.dct(block.T, norm='ortho').T, norm='ortho')

def idct2(block):
    return scipy.fftpack.idct(scipy.fftpack.idct(block.T, norm='ortho').T, norm='ortho')


def quantize(block, quant_matrix):
    return np.round(block / quant_matrix)

def dequantize(block, quant_matrix):
    return block * quant_matrix

# Huffman κωδικοποίηση
class Node:
    def __init__(self, frequency, symbol, left=None, right=None):
        self.frequency = frequency
        self.symbol = symbol
        self.left = left
        self.right = right
        self.huff = ''

    def __lt__(self, other):
        return self.frequency < other.frequency

def calculate_codes(node, val=''):
    codes = {}
    new_val = val + str(node.huff)
    if node.left:
        codes.update(calculate_codes(node.left, new_val))
    if node.right:
        codes.update(calculate_codes(node.right, new_val))
    if not node.left and not node.right:
        codes[node.symbol] = new_val
    return codes

def huffman_encoding(data):
    symbol_freq = Counter(data)
    heap = [Node(freq, symbol) for symbol, freq in symbol_freq.items()]
    heapify(heap)

    while len(heap) > 1:
        left = heappop(heap)
        right = heappop(heap)
        left.huff = '0'
        right.huff = '1'
        new_node = Node(left.frequency + right.frequency, left.symbol + left.symbol + right.symbol, left, right)
        heappush(heap, new_node)

    huffman_tree = heappop(heap)
    huffman_codes = calculate_codes(huffman_tree)

    encoded_data = ''.join([huffman_codes[symbol] for symbol in data])
    return encoded_data, huffman_codes


def calculate_average_code_length(huffman_codes, symbol_freq):
    total_length = sum(len(huffman_codes[symbol]) * freq for symbol, freq in symbol_freq.items())
    total_symbols = sum(symbol_freq.values())
    return total_length / total_symbols


def jpeg_compression_with_huffman(image, quant_matrix):
    height, width = image.shape
    compressed = np.zeros_like(image, dtype=np.float32)
    all_quant_blocks = []

    for i in range(0, height, 8):
        for j in range(0, width, 8):
            block = image[i:i+8, j:j+8]
            dct_block = dct2(block)
            quant_block = quantize(dct_block, quant_matrix)
            compressed[i:i+8, j:j+8] = quant_block
            all_quant_blocks.extend(quant_block.flatten())


    encoded_data, huffman_codes = huffman_encoding(all_quant_blocks)
    symbol_freq = Counter(all_quant_blocks)
    avg_code_length = calculate_average_code_length(huffman_codes, symbol_freq)

    return compressed, encoded_data, avg_code_length

# Αποσυμπίεση JPEG
def jpeg_decompression_with_huffman(compressed, quant_matrix):
    height, width = compressed.shape
    decompressed = np.zeros_like(compressed, dtype=np.float32)

    for i in range(0, height, 8):
        for j in range(0, width, 8):
            quant_block = compressed[i:i+8, j:j+8]
            dequant_block = dequantize(quant_block, quant_matrix)
            block = idct2(dequant_block)
            decompressed[i:i+8, j:j+8] = block

    return decompressed


def scale_quantization_matrix(Q, scale):
    scale_factor = 50 / scale if scale < 50 else 2 - (scale / 50)
    return np.floor((Q * scale_factor + 50) / 100)

# Υπολογισμός PSNR
def calculate_psnr(original, decompressed):
    mse = np.mean((original - decompressed) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr


Q50 = np.array([
    [16, 11, 10, 16, 24, 40, 51, 61],
    [12, 12, 14, 19, 26, 58, 60, 55],
    [14, 13, 16, 24, 40, 57, 69, 56],
    [14, 17, 22, 29, 51, 87, 80, 62],
    [18, 22, 37, 56, 68, 109, 103, 77],
    [24, 35, 55, 64, 81, 104, 113, 92],
    [49, 64, 78, 87, 103, 121, 120, 101],
    [72, 92, 95, 98, 112, 100, 103, 99]
])

Q10 = scale_quantization_matrix(Q50, 10)


image_names = ['/content/gdrive/MyDrive/images-project-2/bridge.bmp',
               '/content/gdrive/MyDrive/images-project-2/girlface.bmp',
               '/content/gdrive/MyDrive/images-project-2/lighthouse.bmp']
quant_matrices = {'Q10': Q10, 'Q50': Q50}


results = {}
for image_name in image_names:
    image = cv2.imread(image_name, cv2.IMREAD_GRAYSCALE)
    results[image_name] = {}
    for qname, quant_matrix in quant_matrices.items():
        compressed, encoded_data, avg_code_length = jpeg_compression_with_huffman(image, quant_matrix)
        decompressed = jpeg_decompression_with_huffman(compressed, quant_matrix)

        original_size = image.size * 8  # σε bits
        compressed_size = len(encoded_data)
        compression_ratio = original_size / compressed_size

        psnr = calculate_psnr(image, decompressed)

        results[image_name][qname] = {
            'avg_code_length': avg_code_length,
            'compression_ratio': compression_ratio,
            'psnr': psnr,
            'decompressed_image': decompressed
        }


for image_name, res in results.items():
    print(f"Results for {image_name}:")
    for qname, metrics in res.items():
        print(f"  Quantization: {qname}")
        print(f"    Average Code Length: {metrics['avg_code_length']:.2f} bits")
        print(f"    Compression Ratio: {metrics['compression_ratio']:.2f}")
        print(f"    PSNR: {metrics['psnr']:.2f} dB")
    print()

    plt.figure(figsize=(12, 8))
    plt.subplot(1, 3, 1)
    plt.imshow(cv2.imread(image_name, cv2.IMREAD_GRAYSCALE), cmap='gray')
    plt.title(f'Original {image_name.split("/")[-1]}')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(res['Q10']['decompressed_image'], cmap='gray')
    plt.title(f'Decompressed Q10\nPSNR: {res["Q10"]["psnr"]:.2f} dB')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(res['Q50']['decompressed_image'], cmap='gray')
    plt.title(f'Decompressed Q50\nPSNR: {res["Q50"]["psnr"]:.2f} dB')
    plt.axis('off')

    plt.show()

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import local_binary_pattern
from skimage.feature import local_binary_pattern

dataset_path = "/content/gdrive/MyDrive/Villains"

categories = ["Darth Vader", "Green Goblin", "Joker", "Thanos", "Venom"]

def load_and_convert_images(dataset_path, categories):
    images = {}
    for category in categories:
        category_path = os.path.join(dataset_path, category)
        images[category] = []
        for filename in os.listdir(category_path):
            if filename.endswith(".jpg") or filename.endswith(".png"):
                image_path = os.path.join(category_path, filename)
                image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                if image is not None:
                    images[category].append(image)
    return images

images = load_and_convert_images(dataset_path, categories)

def extract_normalized_histogram(image, bins=256):
    hist = cv2.calcHist([image], [0], None, [bins], [0, 256])
    hist = hist / hist.sum()
    return hist.flatten()

brightness_histograms = {category: [extract_normalized_histogram(img) for img in images[category]] for category in categories}

radius = 1
n_points = 8 * radius

def extract_lbp_histogram(image, radius, n_points, bins=256):
    lbp = local_binary_pattern(image, n_points, radius, method='uniform')
    hist, _ = np.histogram(lbp, bins=bins, range=(0, bins))
    hist = hist / hist.sum()
    return hist.flatten()

lbp_histograms = {category: [extract_lbp_histogram(img, radius, n_points) for img in images[category]] for category in categories}


def plot_image_and_histograms(image, brightness_histogram, lbp_histogram):
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.imshow(image, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.plot(brightness_histogram)
    plt.title('Normalized Brightness Histogram')

    plt.subplot(1, 3, 3)
    plt.plot(lbp_histogram)
    plt.title('Normalized LBP Histogram')

    plt.show()

for category in categories:
    sample_image = images[category][0]
    brightness_histogram = brightness_histograms[category][0]
    lbp_histogram = lbp_histograms[category][0]
    plot_image_and_histograms(sample_image, brightness_histogram, lbp_histogram)

# Μετρική Β1: Manhattan Distance (Επίπεδο Λ1)
def compute_l1_distance(f1, f2):
    return np.sum(np.abs(f1 - f2))

# Μετρική Β2: Euclidean Distance (Επίπεδο Λ2)
def compute_l2_distance(f1, f2):
    return np.sqrt(np.sum((f1 - f2)**2))

# Παράδειγμα χρήσης:
f1 = np.array([1, 2, 3, 4, 5])
f2 = np.array([5, 4, 3, 2, 1])

l1_distance = compute_l1_distance(f1, f2)
l2_distance = compute_l2_distance(f1, f2)

print("L1 Distance:", l1_distance)
print("L2 Distance:", l2_distance)

from skimage.feature import local_binary_pattern
from sklearn.metrics.pairwise import manhattan_distances, euclidean_distances
import random


# Λίστα με τις κατηγορίες και τις εικόνες ανά κατηγορία
categories = ["Darth Vader", "Green Goblin", "Joker", "Thanos", "Venom"]
images = {category: [] for category in categories}

# Φόρτωση εικόνων από κάθε κατηγορία
for category in categories:
    category_path = os.path.join(dataset_path, category)
    for filename in os.listdir(category_path):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image_path = os.path.join(category_path, filename)
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if image is not None:
                images[category].append(image)

# Τυχαία επιλογή μίας εικόνας από κάθε κατηγορία ως query image
query_images = {category: random.choice(images[category]) for category in categories}

# Συνάρτηση για εξαγωγή LBP χαρακτηριστικών
def extract_lbp_features(image, radius=1, n_points=8):
    lbp = local_binary_pattern(image, n_points, radius, method='uniform')
    hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, n_points + 3), range=(0, n_points + 2))
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-7)  # Κανονικοποίηση
    return hist

# Υπολογισμός LBP χαρακτηριστικών για όλες τις εικόνες στο dataset
lbp_features = {category: [extract_lbp_features(img) for img in images[category]] for category in categories}

# Υπολογισμός μετρικών ομοιότητας για την query image με όλες τις υπόλοιπες εικόνες
results = {}
for category in categories:
    query_feature = extract_lbp_features(query_images[category])
    category_results = []
    for other_category in categories:
        if other_category != category:
            for other_image, other_feature in zip(images[other_category], lbp_features[other_category]):
                l1_distance = manhattan_distances([query_feature], [other_feature])[0][0]
                l2_distance = euclidean_distances([query_feature], [other_feature])[0][0]
                category_results.append((other_category, other_image, l1_distance, l2_distance))
    category_results.sort(key=lambda x: (x[2], x[3]))  # Ταξινόμηση με βάση την L1 και μετά L2 απόσταση
    results[category] = category_results[:5]  # Κρατάμε τα top-5 αποτελέσματα

# Εμφάνιση των top-5 αποτελεσμάτων για κάθε query image
for category in categories:
    print(f"Results for query image from category: {category}")
    for rank, (other_category, image, l1_distance, l2_distance) in enumerate(results[category], 1):
        print(f"  Rank {rank}: Category {other_category}, L1 Distance: {l1_distance:.2f}, L2 Distance: {l2_distance:.2f}")



