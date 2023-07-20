using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System.IO;
using System;
using System.Linq;
public class TimeDecoder : MonoBehaviour
{
    public RenderTexture encodeVideo;
    public ComputeShader computeShader;
    private RenderTexture decodeVideo;

    private List<int> deltaTime = new List<int> { };
    // Start is called before the first frame update
    void Start()
    {
        decodeVideo = new RenderTexture(3840, 1920, 0);
        decodeVideo.enableRandomWrite = true;
        decodeVideo.Create();

    }

    // Update is called once per frame
    void Update()
    {
        RenderTexture.active = encodeVideo;
        Texture2D video2D = new Texture2D(encodeVideo.width, encodeVideo.height, TextureFormat.RGBA32, false);
        video2D.ReadPixels(new Rect(0, 0, encodeVideo.width, encodeVideo.height), 0, 0);
        video2D.Apply();
        RenderTexture.active = null;

        int[] timedecoded = timeDecoder(video2D);
        Debug.Log(timedecoded[0] + ":" + timedecoded[1] + ":" + timedecoded[2] + ":" + timedecoded[3]);
        Destroy(video2D);

        // Latency
        int time_hours = DateTime.Now.TimeOfDay.Hours;
        int time_minutes = DateTime.Now.TimeOfDay.Minutes;
        int time_seconds = DateTime.Now.TimeOfDay.Seconds;
        int time_milliseconds = DateTime.Now.TimeOfDay.Milliseconds;

        deltaTime.Add((time_hours - timedecoded[0]) * 60 * 60 * 1000 + (time_minutes - timedecoded[1]) * 60 * 1000
                    + (time_seconds - timedecoded[2]) * 1000 + (time_milliseconds - timedecoded[3]));

    }

    private void OnApplicationQuit()
    {
        string path = "Assets/Resources/av1.txt";
        StreamWriter writer = new StreamWriter(path, false);

        foreach (int deltat in deltaTime) 
        {
            writer.WriteLine(deltat);
        }
        writer.Close();
    }

    int[] timeDecoder(Texture2D video2D)
    {
        int[] timedecoded = new int[4];
        
        //hours
        int firstDigit = getTimePixel(video2D, 20);
        firstDigit = getTimeNum(firstDigit);
        int secondDigit = getTimePixel(video2D, 80);
        secondDigit = getTimeNum(secondDigit);
        timedecoded[0] = firstDigit * 10 + secondDigit;

        //minutes
        firstDigit = getTimePixel(video2D, 160);
        firstDigit = getTimeNum(firstDigit);
        secondDigit = getTimePixel(video2D, 220);
        secondDigit = getTimeNum(secondDigit);
        timedecoded[1] = firstDigit * 10 + secondDigit;

        //seconds
        firstDigit = getTimePixel(video2D, 300);
        firstDigit = getTimeNum(firstDigit);
        secondDigit = getTimePixel(video2D, 360);
        secondDigit = getTimeNum(secondDigit);
        timedecoded[2] = firstDigit * 10 + secondDigit;

        //milliseconds
        firstDigit = getTimePixel(video2D, 440);
        firstDigit = getTimeNum(firstDigit);
        secondDigit = getTimePixel(video2D, 500);
        secondDigit = getTimeNum(secondDigit);
        int thirdDigit = getTimePixel(video2D, 560);
        thirdDigit = getTimeNum(thirdDigit);
        timedecoded[3] = firstDigit * 100 + secondDigit * 10 + thirdDigit;

        return timedecoded;
    }

    int getTimePixel(Texture2D video2D, int startPixel)
    {
        int encodeTime = 0;
        int[] checkCoords = new int[] { startPixel + 25, 13, 
                                        startPixel + 3,  35, 
                                        startPixel + 48, 35, 
                                        startPixel + 25, 58,
                                        startPixel + 3,  80,
                                        startPixel + 48, 80,
                                        startPixel + 25, 103 };
        double colorMSE = 0;
        Color videoColor;

        for (int i = 0; i < checkCoords.Length; i += 2)
        {
            videoColor = video2D.GetPixel(checkCoords[i], checkCoords[i+1]);
            colorMSE = ( Math.Pow(videoColor.r - 1, 2) + Math.Pow(videoColor.g - 0, 2) 
                       + Math.Pow(videoColor.b - 0, 2) + Math.Pow(videoColor.a - 1, 2)) / 4;
            if (colorMSE < 1e-1)
                encodeTime = encodeTime + (int)Math.Pow(10, i / 2);
        }

        return encodeTime;
    }

    int getTimeNum(int encodeTime)
    {
        int timeNum = 0;
        switch (encodeTime)
        {
            case 1110111:
                timeNum = 0;   
                break;
            case 0100100: 
                timeNum = 1;
                break;
            case 1101011:
                timeNum = 2;
                break;
            case 1101101:
                timeNum = 3;
                break;
            case 0111100:
                timeNum = 4;
                break;
            case 1011101:
                timeNum = 5;
                break;
            case 1011111:
                timeNum = 6;
                break;
            case 1100100:
                timeNum = 7;
                break;
            case 1111111:
                timeNum = 8;
                break;
            case 1111101:
                timeNum = 9;
                break;
        }
        return timeNum;
    }
}
