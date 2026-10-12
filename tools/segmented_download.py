#!/usr/bin/env python3
"""Resumable HTTP range downloader for high-latency dataset mirrors."""

from __future__ import annotations
import argparse, concurrent.futures, os, threading, time, urllib.request
from pathlib import Path


def remote_size(url):
    request = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(request, timeout=60) as response:
        return int(response.headers["Content-Length"]), response.geturl()


def fetch(url, path, start, end, retries):
    path.parent.mkdir(parents=True, exist_ok=True)
    expected = end-start+1
    for attempt in range(retries):
        have = path.stat().st_size if path.exists() else 0
        if have == expected: return expected
        if have > expected:
            path.unlink(); have = 0
        request = urllib.request.Request(url, headers={"Range":f"bytes={start+have}-{end}"})
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                if response.status != 206:
                    raise RuntimeError(f"range request returned HTTP {response.status}")
                with path.open("ab") as output:
                    while True:
                        block=response.read(1024*1024)
                        if not block: break
                        output.write(block)
            if path.stat().st_size == expected: return expected
        except Exception as exc:
            if attempt+1 == retries: raise
            print(f"RETRY {path.name} {attempt+1}/{retries}: {exc}",flush=True)
            time.sleep(min(30,2**attempt))
    raise RuntimeError("unreachable")


def main():
    p=argparse.ArgumentParser()
    p.add_argument("url"); p.add_argument("output",type=Path)
    p.add_argument("--connections",type=int,default=24); p.add_argument("--retries",type=int,default=12)
    a=p.parse_args(); size,final_url=remote_size(a.url)
    parts_dir=Path(str(a.output)+".segments"); parts_dir.mkdir(parents=True,exist_ok=True)
    ranges=[]
    for i in range(a.connections):
        start=size*i//a.connections; end=size*(i+1)//a.connections-1
        ranges.append((parts_dir/f"part_{i:03d}",start,end))
    stop=threading.Event()
    def report():
        while not stop.wait(10):
            have=sum(x[0].stat().st_size for x in ranges if x[0].exists())
            print(f"PROGRESS {have}/{size} {100*have/size:.2f}%",flush=True)
    reporter=threading.Thread(target=report,daemon=True); reporter.start()
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=a.connections) as pool:
            futures=[pool.submit(fetch,final_url,*item,a.retries) for item in ranges]
            for future in concurrent.futures.as_completed(futures): future.result()
    finally:
        stop.set(); reporter.join(timeout=1)
    temporary=Path(str(a.output)+".assembling")
    with temporary.open("wb") as output:
        for path,_,_ in ranges:
            with path.open("rb") as source:
                while True:
                    block=source.read(4*1024*1024)
                    if not block: break
                    output.write(block)
    if temporary.stat().st_size != size: raise RuntimeError("assembled size mismatch")
    os.replace(temporary,a.output)
    print(f"COMPLETE {a.output} {size}",flush=True)


if __name__=="__main__": main()
