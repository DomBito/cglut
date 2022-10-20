install: cglut.py colorlib.py
	install colorlib.py /usr/lib/python3.10
	install cglut.py /usr/bin/cglut

run: cglut.py colorlib.py
	python cglut.py

clean:
	rm -rf __pycache__
