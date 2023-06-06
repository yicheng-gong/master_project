using System.Collections.Generic;
using UnityEngine;
using UnityEngine.XR;

public class CameraFollower : MonoBehaviour
{
    private InputDevice vrDevice;
    private bool isTracking = false;

    void Start()
    {
        // 获取VR设备
        var inputDevices = new List<InputDevice>();
        InputDevices.GetDevicesWithCharacteristics(InputDeviceCharacteristics.HeadMounted, inputDevices);
        if (inputDevices.Count > 0)
        {
            vrDevice = inputDevices[0];
            isTracking = true;
        }
    }

    void Update()
    {
        if (isTracking && vrDevice != null)
        {
            // 获取VR头显的位置和旋转信息
            Vector3 headsetPosition;
            if (vrDevice.TryGetFeatureValue(CommonUsages.devicePosition, out headsetPosition))
            {
                // 更新摄像机位置
                transform.position = headsetPosition;
            }

            Quaternion headsetRotation;
            if (vrDevice.TryGetFeatureValue(CommonUsages.deviceRotation, out headsetRotation))
            {
                // 更新摄像机旋转
                transform.rotation = headsetRotation;
            }
        }
    }
}

