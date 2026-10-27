#!/usr/bin/env bash
set -euo pipefail

gpus=${1:?usage: p2p.sh GPU_LIST}
dwords="ForceP2P=0x11;RMForceP2PType=1;RMPcieP2PType=2;GrdmaPciTopoCheckOverride=1;EnableResizableBar=1"
modprobe_path=/etc/modprobe.d/nvidia-p2p.conf
modprobe_conf="options nvidia NVreg_RegistryDwords=\"$dwords\"
options nvidia_uvm uvm_disable_hmm=1"

die() {
    echo "p2p.sh: $*" >&2
    exit 1
}

slow() {
    echo "$*" >&2
    read -rp "Serve anyway (much slower)? [y/N] " ok
    [[ $ok == [yY]* ]] || exit 1
    exit 0
}

root() {
    if ((EUID)); then sudo "$@"; else "$@"; fi
}

has() {
    [[ " $1 " == *" $2 "* ]]
}

p2p_ok() {
    nvidia-smi topo -p2p "$1" | awk -v gpus="$gpus" '
        BEGIN { n = split(gpus, g, ","); for (i = 1; i <= n; i++) use[g[i]] = 1 }
        /^ *GPU[0-9]/ && substr($1, 4) in use { for (i = 2; i <= NF; i++) if ((i - 2) in use && $i != "X" && $i != "OK") bad = 1 }
        END { exit bad }'
}

[[ -r /proc/driver/nvidia/params ]] || die "the NVIDIA driver isn't loaded"

case $(grep -m1 vendor_id /proc/cpuinfo) in
    *AuthenticAMD*) args=(iommu=off amd_iommu=off) ;;
    *GenuineIntel*) args=(iommu=off intel_iommu=off) ;;
    *) args=(iommu=off) ;;
esac

if command -v kernelstub >/dev/null; then
    boot=kernelstub
elif command -v grubby >/dev/null; then
    boot=grubby
elif [[ -f /etc/default/grub ]] && command -v update-grub >/dev/null; then
    boot=grub
else
    boot=
fi

if command -v update-initramfs >/dev/null; then
    initramfs=(update-initramfs -u)
elif command -v dracut >/dev/null; then
    initramfs=(dracut -f)
else
    initramfs=()
fi

saved_cmdline() {
    case $boot in
        kernelstub) tr -d '",[]' < /etc/kernelstub/configuration | tr -s ' \n' ' ' ;;
        grubby) root grubby --info=DEFAULT | sed -n 's/^args="\(.*\)"$/\1/p' ;;
        grub) (. /etc/default/grub && echo "${GRUB_CMDLINE_LINUX:-} ${GRUB_CMDLINE_LINUX_DEFAULT:-}") ;;
    esac
}

add_args() {
    case $boot in
        kernelstub) root kernelstub -a "$*" ;;
        grubby) root grubby --update-kernel=ALL --args="$*" ;;
        grub) root sed -i "s/^GRUB_CMDLINE_LINUX_DEFAULT=\"/&$* /" /etc/default/grub && root update-grub ;;
    esac
}

live_cmdline=$(< /proc/cmdline)
saved=$(saved_cmdline)
args_live=true
missing=()
for arg in "${args[@]}"; do
    has "$live_cmdline" "$arg" || args_live=false
    has "$saved" "$arg" || missing+=("$arg")
done
[[ -z $(ls -A /sys/kernel/iommu_groups 2>/dev/null) ]] && iommu_off=true || iommu_off=false
live=true
$iommu_off || $args_live || live=false
[[ $(< /proc/driver/nvidia/params) == *"RegistryDwords: \"$dwords\""* ]] || live=false
[[ $(cat /sys/module/nvidia_uvm/parameters/uvm_disable_hmm 2>/dev/null) == Y ]] || live=false
[[ $(cat "$modprobe_path" 2>/dev/null) == "$modprobe_conf" ]] && write_modprobe=false || write_modprobe=true

if ((${#missing[@]})) && [[ -z $boot ]]; then
    echo "p2p.sh: can't find kernelstub, grubby or GRUB; add these to your kernel command line yourself: ${missing[*]}" >&2
    missing=()
fi
if $write_modprobe && ((${#initramfs[@]} == 0)); then
    echo "p2p.sh: can't find update-initramfs or dracut; rebuild your initramfs yourself after this" >&2
fi

plan=()
$write_modprobe && plan+=("write $modprobe_path (NVIDIA driver P2P options)")
((${#missing[@]})) && plan+=("add '${missing[*]}' to the kernel command line ($boot)")
$write_modprobe && ((${#initramfs[@]})) && plan+=("rebuild the initramfs (${initramfs[*]}, takes a minute or two)")

if ((${#plan[@]})); then
    echo "Fast GPU-to-GPU (P2P) transfers need some system settings changed. This will:"
    printf '  - %s\n' "${plan[@]}"
    read -rp "Change them? [y/N] " ok
    [[ $ok == [yY]* ]] || slow "Leaving your system settings alone."
    if $write_modprobe; then
        echo "$modprobe_conf" | root tee "$modprobe_path" >/dev/null
    fi
    ((${#missing[@]} == 0)) || add_args "${missing[@]}"
    if $write_modprobe && ((${#initramfs[@]})); then
        root "${initramfs[@]}"
    fi
fi

$live || slow "The P2P settings are saved but only take effect after a reboot; reboot, then run serve.sh again."
$iommu_off || slow "The IOMMU is still on even with ${args[*]} set; disable it in your BIOS, then run serve.sh again."
for mode in r w; do
    p2p_ok "$mode" || { nvidia-smi topo -p2p "$mode" >&2; slow "The driver doesn't report P2P as OK between GPUs $gpus (see above)."; }
done
echo "P2P is working between GPUs $gpus."
