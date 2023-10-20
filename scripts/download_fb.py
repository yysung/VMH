import re
import bs4
import argparse
import requests

from tqdm import tqdm


parser = argparse.ArgumentParser(description='Facebook Video Downloader')
parser.add_argument('--url', '-u',
                    default=None, type=str, help='Facebook URL that contains video to download')

parser.add_argument('--path', '-p',
                    default=None, type=str, help='path to save the videos')

parser.add_argument('--max_size', '-ms',
                    default=-1, type=int, help='Maximum file size (KB) to download (-1: unlimited)')

parser.add_argument('--max_duration', '-md',
                    default=-1, type=int, help='Maximum video duration (sec) to download (-1: unlimited)')

args = parser.parse_args()
url = args.url

def get_request(video_url):
    return requests.get(video_url, timeout=10)

if args.max_duration > 0:
    import cv2
    
def parse_video_url(url):
    r = requests.get(str(url), timeout=10)
    if r.status_code == 200:
        bs = bs4.BeautifulSoup(r.text, 'html.parser')
        scripts = bs.find_all('meta')
        for s in scripts:
            s = str(s)
            if 'xx.fbcdn.net' in s and '.mp4' in s:
                video_url = re.findall(r'"([^"]*)"', s)[0]
                video_url = video_url.replace('&amp;', '&')
                #print(video_url)
                r.close()
                break
    else:
        video_url = None
        
    return r.status_code, video_url
    
def download_video_from_url(url):
    parse_code, video_url = parse_video_url(url)
    video_name = url.split('/')[3] + '-' + url.split('/')[5] + '.mp4'
    
    if parse_code != 200:
        return video_name, f'Parsing {url} failed (code: {parse_code})'
    
    r = requests.get(video_url, stream=True)
    
    if r.status_code == 200:
        file_size = int(r.headers['Content-Length'])
        if args.max_size > 0:
            if file_size / 1024 > args.max_file_size:
                return video_name, f'File is larger than max file size: {file_size / 1024} > args.max_file_size'
            
        if args.max_duration > 0:
            v = cv2.VideoCapture(video_url)
            frames = v.get(cv2.CAP_PROP_FRAME_COUNT)
            fps = v.get(cv2.CAP_PROP_FPS)
            duration = frames / fps
            
            if duration > args.max_duration:
                return video_name, f'File is longer than max duration: {duration} > args.max_duration'
        
        block_size = 1024
        pbar = tqdm(total=file_size, unit='B', unit_scale=True, desc=video_name, ascii=True)
        with open(video_name, 'wb') as f:
            for data in r.iter_content(block_size):
                pbar.update(len(data))
                f.write(data)
        pbar.close()
        
        r.close()
        return video_name, f'Downloaded Successfully ({file_size/1024:.2f} KB)'
    
    else:
        return video_name, f'Request Failed (code: {r.status_code}'
    
filename, msg = download_video_from_url(url)
print('\n', msg)
