using System.Collections.Generic;
using UnityEngine;
using UnityEngine.XR;

public class CameraFollower : MonoBehaviour
{
    private InputDevice vrDevice;
    private bool isTracking = false;
    public Camera cameraToFollow1;
    public Camera cameraToFollow2;

    void Start()
    {
        // Get VR Device
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
            Quaternion headsetRotation;
            if (vrDevice.TryGetFeatureValue(CommonUsages.deviceRotation, out headsetRotation))
            {
                // uodate direction
                cameraToFollow1.transform.rotation = headsetRotation;
                cameraToFollow2.transform.rotation = headsetRotation;
            }
        }
    }
}

