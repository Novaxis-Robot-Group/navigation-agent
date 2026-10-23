# Sourced by the entrypoint and interactive Bash (including docker exec).
# Prefer the cu130 wheels' userspace libraries over any system CUDA libraries.
for lightnav_lib in /opt/lightnav/lib/python*/site-packages/nvidia/*/lib; do
    if [ -d "$lightnav_lib" ]; then
        case ":${LD_LIBRARY_PATH:-}:" in
            *":$lightnav_lib:"*) ;;
            *) export LD_LIBRARY_PATH="$lightnav_lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" ;;
        esac
    fi
done
unset lightnav_lib
