using System.Collections.Generic;
using UnityEngine;
using UnityEngine.XR;

public class CameraFollower : MonoBehaviour
{
    private InputDevice vrDevice;
    private bool isTracking = false;

    void Start()
    {
        // ��ȡVR�豸
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
            // ��ȡVRͷ�Ե�λ�ú���ת��Ϣ
            Vector3 headsetPosition;
            if (vrDevice.TryGetFeatureValue(CommonUsages.devicePosition, out headsetPosition))
            {
                // ���������λ��
                transform.position = headsetPosition;
            }

            Quaternion headsetRotation;
            if (vrDevice.TryGetFeatureValue(CommonUsages.deviceRotation, out headsetRotation))
            {
                // �����������ת
                transform.rotation = headsetRotation;
            }
        }
    }
}

