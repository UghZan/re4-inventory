using System.Collections;
using System.Collections.Generic;
using UnityEngine;

[CreateAssetMenu(menuName = "Items/Data/New Grid Matrix")]
public class ItemGridMatrix : ScriptableObject
{
    public int sizeX = 2;
    public int sizeY = 2;
    public bool[] occupiedMatrix = new bool[4];

    public void UpdateMatrix()
    {
        occupiedMatrix = new bool[sizeY * sizeX];
    }

    public Vector2Int GetRotatedSize(int rot)
    {
        if (rot == 0 || rot == 2)
            return new Vector2Int(sizeX, sizeY);
        else
            return new Vector2Int(sizeY, sizeX);
    }

    //0 - no rotation
    //1 - 90 degrees clockwise
    //2 - 180 degrees clockwise
    //3 - 270 degrees clockwise
    public bool GetAtRotated(int rot, int x, int y)
    {
        switch (rot)
        {
            //dark magic
            case 0:
            default:
                return occupiedMatrix[y * sizeX + x];
            case 1:
                return occupiedMatrix[(sizeY - x - 1) * sizeX + y];
            case 2:
                return occupiedMatrix[(sizeX - 1 - x) + (sizeY - y - 1) * sizeX];
            case 3:
                return occupiedMatrix[(x+1) * sizeX - 1 - y];
        }
    }

    public void Print()
    {
        string printout = "";
        for (int i = 0; i < 4; i++)
        {
            Vector2Int size = GetRotatedSize(i);
            Debug.Log(size);
            printout += $"Rotated by {i * 90} degrees clockwise:\n";
            for (int j = 0; j < size.y; j++)
            {
                for (int k = 0; k < size.x; k++)
                {
                    Debug.Log($"iteration {i} {j} {k}");
                    printout += (GetAtRotated(i, k, j) ? "X " : "O ");
                }
                printout += "\n";
            }
            Debug.Log("rotation " + i);
        }


        Debug.Log(printout);
    }

}
