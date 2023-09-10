using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.SceneManagement;
using UnityEngine.XR;
public class RechooseHQ : MonoBehaviour
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
            if (devices[i].TryGetFeatureValue(CommonUsages.menuButton, out bool menuButtonValue) && menuButtonValue)
            {
                Debug.Log("Menu button was pressed.");
                Invoke(nameof(LoadStart), 3);
            }
        }
    }

    void LoadStart()
    {
        SceneManager.LoadScene("Start");
    }
}
