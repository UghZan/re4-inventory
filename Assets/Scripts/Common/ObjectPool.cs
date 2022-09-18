using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Events;

//Object Pooling implementation
public class ObjectPool : MonoBehaviour
{
    [SerializeField] GameObject _poolObject;
    [SerializeField] int _preSpawns;//amount at start
    [SerializeField] int _overflowSpawnAmount; //in case we need more, how much additional objects to spawn
    public UnityEvent OnObjectDisabled;
    public int ObjectsActive;

    int _currentAmount;
    void Start()
    {
        for (int i = 0; i < _preSpawns; i++)
        {
            Instantiate(_poolObject, transform.position, Quaternion.identity, transform).SetActive(false);
            _currentAmount++;
        }
        OnObjectDisabled.AddListener(() => ObjectsActive--);
    }

    public void ResetPool()
    {
        for (int i = 0; i < _currentAmount; i++)
        {
            transform.GetChild(i).gameObject.SetActive(false);
        }
        ObjectsActive = 0;
    }

    public GameObject GetPooledObject()
    {
        for (int i = 0; i < _currentAmount; i++)
        {
            if (!transform.GetChild(i).gameObject.activeInHierarchy)
            {
                ObjectsActive++;
                return transform.GetChild(i).gameObject;
            }
        }
        //if we are out of available objects, expand the pool
        int previous = _currentAmount;
        for (int i = 0; i < _overflowSpawnAmount; i++)
        {
            Instantiate(_poolObject, transform.position, Quaternion.identity, transform).SetActive(false);
            _currentAmount++;
        }
        for (int i = previous; i < _currentAmount; i++)
        {
            if (!transform.GetChild(i).gameObject.activeInHierarchy)
            {
                ObjectsActive++;
                return transform.GetChild(i).gameObject;
            }
        }
        return null;
    }
}
