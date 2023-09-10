using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.XR;

public class HQCamFollow : MonoBehaviour
{
    private InputDevice vrDevice;
    private bool isTracking = false;
    // Start is called before the first frame update
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

    // Update is called once per frame
    void Update()
    {
        if (isTracking && vrDevice != null)
        {
            Quaternion headsetRotation;
            if (vrDevice.TryGetFeatureValue(CommonUsages.deviceRotation, out headsetRotation))
            {
                // uodate direction
                transform.rotation = headsetRotation;
            }
        }
    }
}
