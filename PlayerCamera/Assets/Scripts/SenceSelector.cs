using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.SceneManagement;
using UnityEngine.XR;

public class SenceSelector : MonoBehaviour
{
    private List<InputDevice> devices;
    // Start is called before the first frame update
    void Start()
    {
        devices = new List<InputDevice>();
    }

    // Update is called once per frame
    void Update()
    {
        InputDevices.GetDevicesWithCharacteristics(InputDeviceCharacteristics.Controller, devices);
        for (int i = 0; i < devices.Count; i++)
        {
            if (devices[i].TryGetFeatureValue(CommonUsages.primaryButton, out bool primaryButtonValue) && primaryButtonValue)
            {
                Debug.Log("X button was pressed.");
                Invoke(nameof(LoadHQ), 3);
            }
            if (devices[i].TryGetFeatureValue(CommonUsages.secondaryButton, out bool secondaryButtonValue) && secondaryButtonValue)
            {
                Debug.Log("Y button was pressed.");
                Invoke(nameof(LoadAHQ), 3);
            }
        }
    }

    void LoadHQ()
    {
        SceneManager.LoadScene("HQ");
    }

    void LoadAHQ()
    {
        SceneManager.LoadScene("AHQ");
    }
}
