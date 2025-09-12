# add hooks when app start, stop.
# apps can add custom actions in these functions, default are empty.

# before application server start, jvm process not exists.
beforeStartApp() {
  echo "beforeStartApp"
	return
}

# after application server start, localhost:8080 is ready.
afterStartApp() {
  echo "afterStartApp"
	return
}

# before application server stop, localhost:8080 is available.
beforeStopApp() {
  echo "beforeStopApp"
	return
}

# after application server stop, jvm process has exited.
afterStopApp() {
  echo "afterStopApp"
	return
}
